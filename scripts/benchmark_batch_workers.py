"""遍历 batch/worker 组合，复用正式 worker 和参考集 benchmark。"""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supership.inference_config import INFERENCE_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--workers', nargs='+', type=int, default=[0, 2, 4, 8])
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', default='configs/dfine/dfine_hgnetv2_s_supership.yml')
    parser.add_argument('--weights', default='weights/baseline.pth')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--skip-accuracy', action='store_true')
    parser.add_argument('--init-timeout', type=float, default=300)
    parser.add_argument('--run-timeout', type=float, default=3600)
    args = parser.parse_args()
    if args.runs < 1 or min(args.batch_sizes) < 1 or min(args.workers) < 0:
        parser.error('runs/batch 必须为正整数，workers 必须非负')
    os.chdir(ROOT)
    INFERENCE_CONFIG.validate()
    if INFERENCE_CONFIG.backend == 'tensorrt':
        engine = Path(INFERENCE_CONFIG.engine_path)
        if not engine.is_file() or not Path(str(engine) + '.json').is_file():
            parser.error('缺少 TensorRT 引擎或元数据，请先运行 scripts/build_tensorrt.py')
        metadata = json.loads(Path(str(engine) + '.json').read_text())
        if max(args.batch_sizes) > metadata['max_batch_size']:
            parser.error(f"batch 超过引擎上限 {metadata['max_batch_size']}，请用 --max-batch-size 重建引擎")
    config = Path(args.config).resolve()
    if not config.is_file():
        parser.error(f'配置不存在: {config}')
    destination = (args.output_dir or Path('output') / datetime.now().strftime('batch-workers-%Y%m%d-%H%M%S-%f')).resolve()
    # 禁止复用目录，避免覆盖之前的检测输出和测量结果。
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'settings.json').write_text(json.dumps({
        **vars(args), 'output_dir': str(destination),
        'inference_config': vars(INFERENCE_CONFIG), 'python': sys.executable,
    }, ensure_ascii=False, indent=2) + '\n')
    rows = []
    for batch in dict.fromkeys(args.batch_sizes):
        for workers in dict.fromkeys(args.workers):
            folder = destination / f'batch-{batch}_workers-{workers}'
            folder.mkdir()
            # 通过 include 继承原配置，仅覆盖 worker，保留生产推理路径。
            overlay = folder / 'config.yml'
            overlay.write_text(f'__include__: [{json.dumps(str(config))}]\nval_dataloader:\n  num_workers: {workers}\n')
            command = [sys.executable, 'supership/benchmark_submission.py',
                       '--image-dir', 'data/SuperShip/reference/images',
                       '--meta-file', 'data/SuperShip/reference/reference_images_meta.json',
                       '--output-file', str(folder / 'submission.json'),
                       '--runs', str(args.runs), '--warmup',
                       '--init-timeout', str(args.init_timeout), '--run-timeout', str(args.run_timeout)]
            if not args.skip_accuracy:
                command += ['--accuracy-gt', 'data/SuperShip/reference/annotations/instances_reference.json']
            command += ['--', sys.executable, 'supership/generate_submission.py', '--worker',
                        '--config', str(overlay), '--weights', args.weights,
                        '--device', args.device, '--batch-size', str(batch)]
            print(f'\n=== batch={batch}, workers={workers} ===', flush=True)
            report = None
            with (folder / 'benchmark.log').open('w') as log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                        try:
                            value = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(value, dict) and 'runs_seconds' in value and 'median_seconds' in value:
                            report = value
                    code = process.wait()
            row = {'batch_size': batch, 'num_workers': workers,
                   'status': 'ok' if code == 0 and report else 'failed',
                   'median_seconds': report['median_seconds'] if code == 0 and report else None,
                   'runs_seconds': report['runs_seconds'] if code == 0 and report else [],
                   'log': str(folder / 'benchmark.log')}
            rows.append(row)
            # 每组完成即落盘，即使后续中断也保留已完成的测量。
            (destination / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
            with (destination / 'summary.csv').open('w', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=list(row))
                writer.writeheader()
                writer.writerows({**item, 'runs_seconds': json.dumps(item['runs_seconds'])} for item in rows)
    print('\nbatch\tworkers\tmedian(s)\tstatus')
    for row in sorted(rows, key=lambda item: item['median_seconds'] if item['median_seconds'] is not None else float('inf')):
        median = row['median_seconds']
        print(f"{row['batch_size']}\t{row['num_workers']}\t{median if median is not None else '-'}\t{row['status']}")
    print(f'结果目录: {destination}')
    return int(any(row['status'] != 'ok' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
