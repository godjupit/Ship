"""Draw the current inference pipeline from saved profiling measurements."""
import argparse
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def draw(metrics, out_dir):
    median = metrics['median_end_to_end_s']
    run = min(metrics['runs'], key=lambda r: abs(r['end_to_end_s'] - median))
    batch = run['batch_size']
    workers = run['num_workers']
    mib = run['h2d_bytes'] / 1024**2
    per_batch = mib / run['batches'] if run['batches'] else 0
    sec = lambda key: f"{run.get(key, 0):.4f} 秒"
    gpu_suffix = '_gpu_s' if str(metrics['device']).startswith('cuda') else '_wall_s'
    chunks = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="1010" viewBox="0 0 1300 1010" role="img" aria-labelledby="title desc">
<title id="title">SuperShip 当前推理数据流水线</title>
<desc id="desc">按当前代码绘制的执行顺序，标注实测累计时间。CPU worker 可预取后续 batch，主进程依次完成传输、模型计算和结果整理。不是逐 batch 的真实时间轨迹。</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#64748b"/></marker></defs>
<rect width="1300" height="1010" fill="#fff"/>
<g font-family="PingFang SC, Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#172b4d">''']

    def text(x, y, label, size=17, color='#172b4d', center=False, weight=400):
        anchor = 'middle' if center else 'start'
        chunks.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(str(label))}</text>')

    def rect(x, y, w, h, fill='#f8fafc', stroke='#dbe2ea', radius=12):
        chunks.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def box(x, y, lines, fill='#fff', w=200, h=100):
        rect(x, y, w, h, fill)
        for i, line in enumerate(lines):
            text(x+w/2, y+30+i*25, line, 17 if i==0 else 15, center=True, weight=500 if i==0 else 400)

    def arrow(path, dashed=False):
        dash = ' stroke-dasharray="6 5"' if dashed else ''
        chunks.append(f'<path d="{path}" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"{dash}/>')

    text(40, 50, 'SuperShip · 当前数据流水线', 29, weight=500)
    text(40, 85, f"{run['images']} 张图像 · batch_size={batch} · {run['batches']} 个 batch · {metrics['device']} · 端到端 {sec('end_to_end_s')}", 18, '#52627a')
    rect(40, 120, 1220, 235, '#f0f6ff', '#d8e6fa')
    text(62, 153, f'CPU 后台生产线：{workers} 个 DataLoader worker 并行处理不同 batch', 20, weight=500)
    box(60, 185, ['读取图像文件', '磁盘 / 文件系统', '按元数据找到图片'])
    box(300, 185, ['JPEG 解码', 'PIL 读取并转换 RGB', '仍在 CPU 内存'])
    box(540, 185, ['缩放图像', 'Resize → 640 × 640', 'CPU 上处理'])
    box(780, 185, ['转换与归一化', 'float32，再除以 255', 'CPU 上完成'])
    box(1020, 185, ['拼成一个 batch', f'[{batch}, 3, 640, 640]', f'约 {per_batch:.2f} MiB / 批'])
    for x in (260, 500, 740, 980):
        arrow(f'M{x} 235 H{x+35}')
    text(62, 326, '后台准备下一批可以与 GPU 计算重叠；worker 的总工作时间不等于主进程等待时间。', 16, '#52627a')
    arrow('M1120 285 V386 H160 V465', True)
    text(525, 378, '预取 / 就绪 batch 缓冲', 16, '#52627a')

    rect(40, 414, 1220, 280, '#fbfcfe', '#dbe2ea')
    text(62, 448, '主进程消费线：每个 batch 按这个顺序执行', 20, weight=500)
    box(60, 475, ['取下一个 batch', '等待已准备的数据', sec('loader_wait_wall_s')])
    box(300, 475, ['CPU → GPU', 'samples.to(device)', sec('h2d'+gpu_suffix)], '#fff5e7')
    box(540, 475, ['GPU 模型计算', 'model(samples)', sec('forward'+gpu_suffix)], '#edf7f2')
    box(780, 475, ['GPU 后处理', '选框、分类、坐标转换', sec('postprocess'+gpu_suffix)], '#edf7f2')
    box(1020, 475, ['回传与 CPU 整理', '.tolist() → Python 字典', sec('results_wall_s')])
    for x in (260, 500, 740, 980):
        arrow(f'M{x} 525 H{x+35}')
    arrow('M1120 575 V645 H160 V580')
    text(620, 630, f'完成本批后再取下一批，循环 {run["batches"]} 次', 17, center=True)
    arrow('M1220 525 H1240 V725 H1060 V765')
    text(1090, 718, '全部 batch 完成', 16, center=True)

    box(900, 770, ['一次性写出全部结果', 'JSON → flush → fsync', sec('write_wall_s')], '#f2f0fa', w=320)
    text(60, 760, '当前实现的几个关键点', 20, weight=500)
    text(60, 797, f'① CPU 先生成 float32：总输入传输约 {mib:.2f} MiB。', 17)
    text(60, 830, f'② pin_memory={run.get("pin_memory", False)}；未使用独立拷贝流。', 17)
    text(60, 863, '③ 主循环按批次推进，结果先累积，最后统一写盘。', 17)

    chunks.append('<path d="M40 911 H1260" stroke="#dbe2ea"/>')
    text(40, 945, '读图方式：箭头表示代码执行关系；节点时间是所选轮次全部 batch 的累计测量。', 16, '#52627a')
    text(40, 975, '这不是真实时间轨迹；CPU 等待可能与 GPU 阶段重叠，不能把各节点时间直接相加。', 16, '#52627a')
    chunks.append('</g></svg>')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'pipeline.svg').write_text('\n'.join(chunks)+'\n', encoding='utf-8')
    diagram = f'''flowchart TD
    A[图像文件] --> B[CPU worker × {workers}：RGB 解码]
    B --> C[CPU：Resize 640×640]
    C --> D[CPU：float32 / 255]
    D --> E[预取 batch 缓冲]
    E --> F[CPU→GPU：{sec('h2d'+gpu_suffix)}]
    F --> G[模型前向：{sec('forward'+gpu_suffix)}]
    G --> H[GPU 后处理：{sec('postprocess'+gpu_suffix)}]
    H --> I[回传与 CPU 整理：{sec('results_wall_s')}]
    I -->|还有 batch| E
    I -->|全部完成| J[统一 JSON 写盘：{sec('write_wall_s')}]
'''
    (out_dir/'pipeline.mmd').write_text(diagram, encoding='utf-8')
    note = f'''# 当前数据执行情况

![当前流水线](pipeline.svg)

- 本轮：{run['images']} 张图像，{run['batches']} 批，batch_size={batch}，端到端 {sec('end_to_end_s')}。
- CPU 有 {workers} 个 DataLoader worker，能后台准备后续批次；解码、缩放、float32 转换和归一化均在 CPU 完成。
- 主进程每次取一批，再传到 GPU，执行前向和后处理，调用 `.tolist()` 将结果回传 CPU 并整理；处理完本批才继续下一批。
- 所有批次处理完成后，统一将结果序列化为 JSON 并刷盘。
- 输入传输总量 {mib:.2f} MiB，平均每批约 {per_batch:.2f} MiB。`pin_memory={run.get('pin_memory', False)}`。
- 主进程等待数据累计 {sec('loader_wait_wall_s')}，不代表全部 CPU worker 工作时间；结果整理时间包含等待 GPU 完成的部分。
- 图中时间是累计值，箭头是根据代码确定的执行关系，不是采集到的逐批次时间线，不能用来判断每个时刻 GPU 是否空闲。
- 多轮时，选择端到端时间最接近中位数的一轮标注；原始测量在 `profile_metrics.json`，文字报告在 `profile_report.txt`。
'''
    (out_dir/'pipeline.md').write_text(note, encoding='utf-8')
    print(f'流水线图：{out_dir / "pipeline.svg"}')
    print(f'分析说明：{out_dir / "pipeline.md"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', default='analysis/profile_metrics.json')
    parser.add_argument('--out-dir', default='analysis')
    args = parser.parse_args()
    metrics_path = Path(args.metrics)
    out_dir = Path(args.out_dir)
    if not metrics_path.is_absolute(): metrics_path = ROOT / metrics_path
    if not out_dir.is_absolute(): out_dir = ROOT / out_dir
    if not metrics_path.exists():
        parser.error('请先运行 python scripts/profile_inference.py 生成计时数据')
    draw(json.loads(metrics_path.read_text(encoding='utf-8')), out_dir)


if __name__ == '__main__':
    main()
