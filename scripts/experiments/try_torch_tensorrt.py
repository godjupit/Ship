"""独立测试 Torch-TensorRT 编译、合成输入误差和稳定前向耗时，不修改正式后端。"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from supership.generate_submission import build_runtime, run_inference
from supership.tensorrt_runner import model_fingerprint


def measure(model, sample, warmup, runs):
    for _ in range(warmup):
        model(sample)
    torch.cuda.synchronize(sample.device)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model(sample)
        torch.cuda.synchronize(sample.device)
        times.append(time.perf_counter() - start)
    return {'runs_seconds': times, 'median_seconds': statistics.median(times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--optimization-level', type=int, choices=range(6), default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', default='configs/dfine/dfine_hgnetv2_s_supership.yml')
    parser.add_argument('--weights', default='weights/baseline.pth')
    parser.add_argument('--output', type=Path, default=Path('output/torch-tensorrt/experiment.json'))
    parser.add_argument('--artifact', type=Path, help='保存或读取的固定 batch 编译模型 .pt2')
    parser.add_argument('--load', action='store_true', help='加载已有产物，不重新编译')
    parser.add_argument('--reference', action='store_true', help='用参考集比较 PyTorch 与编译模型的 AP50')
    args = parser.parse_args()
    if min(args.batch_size, args.runs, args.warmup) < 1:
        parser.error('batch-size/runs/warmup must be positive')
    import torch_tensorrt
    import tensorrt
    torch.cuda.set_device(args.device)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, model, postprocessor = build_runtime(args.config, args.weights, args.device, backend='pytorch')
    sample = torch.rand(args.batch_size, 3, 640, 640, device=args.device)
    report = {'torch': torch.__version__, 'torch_tensorrt': torch_tensorrt.__version__,
              'tensorrt': tensorrt.__version__, 'gpu': torch.cuda.get_device_name(args.device),
              'batch_size': args.batch_size, 'precision': 'fp16' if args.fp16 else 'fp32',
              'tf32': False, 'warmup': args.warmup, 'runs': args.runs,
              'optimization_level': args.optimization_level,
              'scope': 'synthetic fixed-batch forward only; no AP or end-to-end validation'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = args.artifact or args.output.with_suffix('.pt2')
    metadata = {key: report[key] for key in ('torch', 'torch_tensorrt', 'tensorrt', 'gpu',
                                            'batch_size', 'precision', 'optimization_level')}
    metadata['model_fingerprint'] = model_fingerprint(cfg.yaml_cfg, args.weights)
    with torch.inference_mode():
        print('Measuring PyTorch forward...', flush=True)
        expected = model(sample)
        report['pytorch'] = measure(model, sample, args.warmup, args.runs)
        if args.load:
            saved = json.loads(Path(str(artifact) + '.json').read_text())
            if saved != metadata:
                raise ValueError('Artifact metadata differs from model, batch, precision, GPU or runtime; rebuild')
        else:
            started = time.perf_counter()
            # 严格 Dynamo 导出无法正确处理当前编码器的 list.extend(generator)。
            # 非严格导出保留原模型计算，避免为实验改写生产模型。
            print('Exporting model with strict=False...', flush=True)
            exported = torch.export.export(model, (sample,), strict=False)
            exported_output = exported.module()(sample)
            for name in ('pred_logits', 'pred_boxes'):
                torch.testing.assert_close(exported_output[name], expected[name])
            print('Compiling TensorRT partitions...', flush=True)
            compiled = torch_tensorrt.dynamo.compile(
                exported, inputs=[sample],
                enabled_precisions={torch.float32, torch.float16} if args.fp16 else {torch.float32},
                min_block_size=5, disable_tf32=True, use_python_runtime=False,
                pass_through_build_failures=True, device=torch.device(args.device),
                workspace_size=1024**3, max_aux_streams=0,
                optimization_level=args.optimization_level,
            )
            compiled(sample)
            torch.cuda.synchronize(args.device)
            report['compile_seconds'] = time.perf_counter() - started
            artifact.parent.mkdir(parents=True, exist_ok=True)
            print(f'Saving compiled model to {artifact}...', flush=True)
            torch_tensorrt.save(compiled, str(artifact), output_format='exported_program')
            Path(str(artifact) + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print('Loading serialized model...', flush=True)
        compiled = torch.export.load(artifact).module()
        report['artifact'] = str(artifact.resolve())
        print('Checking compiled outputs...', flush=True)
        actual = compiled(sample)
        torch.cuda.synchronize(args.device)
        report['errors'] = {}
        for name in ('pred_logits', 'pred_boxes'):
            delta = (actual[name].float() - expected[name].float()).abs()
            report['errors'][name] = {'max_abs': delta.max().item(), 'mean_abs': delta.mean().item(),
                                      'finite': bool(torch.isfinite(actual[name]).all().item())}
        report['torch_tensorrt_forward'] = measure(compiled, sample, args.warmup, args.runs)
        report['trt_partitions'] = [node.name for node in compiled.graph.nodes
                                    if 'tensorrt.execute_engine' in str(node.target)]
        if not report['trt_partitions']:
            raise RuntimeError('No TensorRT partition found; refusing to report a PyTorch fallback as TensorRT')
        if not all(item['finite'] for item in report['errors'].values()):
            raise RuntimeError('Compiled output contains NaN or infinity')
        if args.reference:
            from supership.compare_reference import score_predictions
            meta_path = 'data/SuperShip/reference/reference_images_meta.json'
            image_count = len(json.loads(Path(meta_path).read_text())['images'])
            if image_count % args.batch_size:
                raise ValueError('Fixed-batch artifact requires reference image count divisible by batch-size')
            report['reference'] = {}
            for name, runner in [('pytorch', model), ('torch_tensorrt', compiled)]:
                print(f'Running reference set with {name}...', flush=True)
                submission = args.output.with_name(f'{args.output.stem}.{name}.submission.json')
                count = run_inference(cfg, runner, postprocessor, args.device, args.batch_size,
                                      'data/SuperShip/reference/images', meta_path, submission)
                metrics = score_predictions(str(submission),
                                           'data/SuperShip/reference/annotations/instances_reference.json')
                report['reference'][name] = {'detections': count, 'AP50': metrics['AP50'],
                                              'submission': str(submission)}
            report['scope'] = 'fixed-batch forward and reference AP50; no end-to-end benchmark or partial-batch validation'
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
