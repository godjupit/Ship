"""Analyze real Nsight Systems SQLite events; never synthesize timestamps."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics


def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def duration(intervals):
    return sum(b-a for a, b in merge(intervals))


def overlap(left, right):
    a, b = merge(left), merge(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1])-max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def stats(values):
    x = sorted(values)
    if not x:
        return dict(count=0, sum_ms=0, mean_ms=0, median_ms=0, p95_ms=0, max_ms=0)
    return dict(count=len(x), sum_ms=sum(x)/1e6, mean_ms=statistics.mean(x)/1e6,
                median_ms=statistics.median(x)/1e6,
                p95_ms=x[min(len(x)-1, int(.95*(len(x)-1)))]/1e6, max_ms=max(x)/1e6)


def csv_file(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(db, out):
    out.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    strings = dict(c.execute('SELECT id,value FROM StringIds'))
    nvtx = []
    for row in c.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        r = dict(row)
        name = r['text'] or strings.get(r['textId'], '')
        if not name:
            continue
        m = re.search(r'/batch=(\d+)', name)
        r.update(name=name, stage=name.split('/')[0],
                 batch=int(m.group(1)) if m else None,
                 pid=(r['globalTid'] >> 24) & 0xffffff,
                 tid=r['globalTid'] & 0xffffff)
        nvtx.append(r)
    inference, = [r for r in nvtx if r['stage'] == 'inference']
    lo, hi = inference['start'], inference['end']
    main_tid = inference['globalTid']
    nvtx = [r for r in nvtx if r['start'] >= lo and r['end'] <= hi]
    api = [dict(r) for r in c.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
    for r in api:
        r['name'] = strings[r['nameId']]
    correlations = {(r['globalTid'] & ~0xffffff, r['correlationId']): r for r in api}
    by_thread = defaultdict(list)
    for r in nvtx:
        by_thread[r['globalTid']].append(r)
    starts = {}
    for tid, rows in by_thread.items():
        rows.sort(key=lambda r: r['start'])
        starts[tid] = [r['start'] for r in rows]

    def parent(r):
        rows = by_thread.get(r['globalTid'], [])
        ix = bisect_right(starts.get(r['globalTid'], []), r['start'])
        for n in reversed(rows[:ix]):
            if n['end'] >= r['end']:
                return n
        return None

    ops = []
    tables = [('CUPTI_ACTIVITY_KIND_KERNEL', 'Kernel'),
              ('CUPTI_ACTIVITY_KIND_MEMCPY', 'Memcpy'),
              ('CUPTI_ACTIVITY_KIND_MEMSET', 'Memset')]
    for table, kind in tables:
        for row in c.execute(f'SELECT * FROM {table} WHERE start>=? AND end<=?', (lo, hi)):
            r = dict(row)
            r['kind'] = kind
            r['name'] = strings.get(r.get('demangledName'), kind)
            if kind == 'Memcpy':
                r['kind'] = {1: 'H2D', 2: 'D2H', 8: 'D2D'}.get(r['copyKind'], f'Memcpy{r["copyKind"]}')
                r['name'] = r['kind']
            launch = correlations.get((r['globalPid'], r['correlationId']))
            n = parent(launch) if launch else None
            r.update(stage=n['stage'] if n else 'unattributed',
                     batch=n['batch'] if n else None,
                     api=launch['name'] if launch else None,
                     api_start=launch['start'] if launch else None,
                     api_end=launch['end'] if launch else None)
            ops.append(r)
    ops.sort(key=lambda r: r['start'])
    assert len([r for r in nvtx if r['stage'] == 'batch_consume_total']) == 25
    assert len([r for r in nvtx if r['stage'] == 'jpeg_read_decode']) == 200
    assert all(r['stage'] != 'unattributed' for r in ops), Counter(r['stage'] for r in ops)

    cpu_stats = []
    for stage in sorted({r['stage'] for r in nvtx}):
        rows = [r for r in nvtx if r['stage'] == stage]
        cpu_stats.append(dict(stage=stage, **stats([r['end']-r['start'] for r in rows]),
                              union_ms=duration([(r['start'], r['end']) for r in rows])/1e6,
                              processes=len({r['pid'] for r in rows})))
    csv_file(out/'cpu_nvtx_stages.csv', cpu_stats)

    gpu_stats = []
    for stage, kind in sorted({(r['stage'], r['kind']) for r in ops}):
        rows = [r for r in ops if r['stage'] == stage and r['kind'] == kind]
        gpu_stats.append(dict(stage=stage, operation=kind,
                              **stats([r['end']-r['start'] for r in rows]),
                              union_ms=duration([(r['start'], r['end']) for r in rows])/1e6,
                              bytes=sum(r.get('bytes', 0) for r in rows),
                              streams=','.join(map(str, sorted({r['streamId'] for r in rows})))))
    csv_file(out/'gpu_actual_stages.csv', gpu_stats)
    csv_file(out/'gpu_events.csv', [dict(start_ns=r['start'], end_ns=r['end'],
        relative_start_ms=(r['start']-lo)/1e6, duration_us=(r['end']-r['start'])/1e3,
        kind=r['kind'], stage=r['stage'], batch=r['batch'], stream=r['streamId'],
        bytes=r.get('bytes', 0), correlation_id=r['correlationId'],
        api=r['api'], api_start_ns=r['api_start'], api_end_ns=r['api_end'],
        kernel_name=r['name']) for r in ops])
    csv_file(out/'cpu_nvtx_events.csv', [dict(start_ns=r['start'], end_ns=r['end'],
        relative_start_ms=(r['start']-lo)/1e6, duration_ms=(r['end']-r['start'])/1e6,
        stage=r['stage'], batch=r['batch'], pid=r['pid'], tid=r['tid'], name=r['name'])
        for r in nvtx])

    def iv(rows):
        return [(r['start'], r['end']) for r in rows]
    kernels = iv([r for r in ops if r['kind'] == 'Kernel'])
    h2d = iv([r for r in ops if r['kind'] == 'H2D'])
    decode = iv([r for r in nvtx if r['stage'] == 'jpeg_read_decode'])
    preprocess = iv([r for r in nvtx if r['stage'] == 'cpu_preprocess'])
    overlap_rows = []
    for label, left, right in [('CPU JPEG vs GPU kernels', decode, kernels),
                               ('CPU preprocessing vs GPU kernels', preprocess, kernels),
                               ('GPU H2D copy vs GPU kernels', h2d, kernels)]:
        overlap_rows.append(dict(pair=label, left_union_ms=duration(left)/1e6,
                                 right_union_ms=duration(right)/1e6,
                                 overlap_ms=overlap(left, right)/1e6,
                                 overlap_of_left_pct=100*overlap(left, right)/duration(left) if duration(left) else 0))
    csv_file(out/'overlap.csv', overlap_rows)

    union = merge(iv(ops))
    gaps = []
    leaf = {'loader_create','loader_start_workers','loader_wait','h2d_submit',
            'gpu_forward_eager_submit','gpu_postprocess_submit','d2h','results_format',
            'loader_finish_workers','existing_final_cuda_sync','output_write'}
    main_rows = [r for r in nvtx if r['globalTid'] == main_tid and r['stage'] in leaf]
    for a, b in zip(union, union[1:]):
        if b[0] <= a[1]:
            continue
        start, end = a[1], b[0]
        matched = [(max(0, min(end,r['end'])-max(start,r['start'])), r) for r in main_rows]
        _, dominant = max(matched, key=lambda pair: pair[0])
        gaps.append(dict(start_ns=start, end_ns=end, start_ms=(start-lo)/1e6,
                         duration_ms=(end-start)/1e6, dominant_cpu_stage=dominant['stage'],
                         batch=dominant['batch']))
    gaps.sort(key=lambda r: -r['duration_ms'])
    csv_file(out/'gpu_idle_gaps.csv', gaps)

    api_in = [r for r in api if r['start'] >= lo and r['end'] <= hi]
    syncs = [r for r in api_in if 'Synchronize' in r['name']]
    sync_rows = []
    for r in syncs:
        n = parent(r)
        sync_rows.append(dict(api=r['name'], stage=n['stage'] if n else 'unattributed',
                              batch=n['batch'] if n else None,
                              start_ms=(r['start']-lo)/1e6, duration_ms=(r['end']-r['start'])/1e6))
    csv_file(out/'cuda_sync_waits.csv', sync_rows)
    sync_stats = []
    for api_name, stage in sorted({(r['api'],r['stage']) for r in sync_rows}):
        rows = [r for r in sync_rows if r['api']==api_name and r['stage']==stage]
        sync_stats.append(dict(api=api_name, stage=stage,
                               **stats([r['duration_ms']*1e6 for r in rows])))
    csv_file(out/'cuda_sync_summary.csv', sync_stats)

    batches = []
    for batch in range(25):
        row = dict(batch=batch)
        for stage in sorted(leaf):
            row[stage+'_cpu_ms'] = sum(r['end']-r['start'] for r in nvtx if r['batch']==batch and r['stage']==stage)/1e6
        for kind in ('H2D','Kernel','D2H','Memset','D2D'):
            row[kind+'_gpu_ms'] = sum(r['end']-r['start'] for r in ops if r['batch']==batch and r['kind']==kind)/1e6
        batches.append(row)
    csv_file(out/'per_batch.csv', batches)
    main_critical = [dict(stage=stage, sum_ms=sum(r['end']-r['start'] for r in main_rows if r['stage']==stage)/1e6)
                     for stage in sorted(leaf)]
    csv_file(out/'main_thread_critical_path.csv', main_critical)
    graph_apis = [r for r in api_in if 'GraphLaunch' in r['name']]
    diagnostic = [dict(r) for r in c.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
    summary = dict(
        source_sqlite=str(db), inference_start_ns=lo, inference_end_ns=hi,
        inference_wall_ms=(hi-lo)/1e6, main_pid=inference['pid'], main_tid=inference['tid'],
        worker_pids=sorted({r['pid'] for r in nvtx if r['stage']=='jpeg_read_decode'}),
        cpu_nvtx=cpu_stats, gpu_actual=gpu_stats, overlap=overlap_rows,
        kernel_count=len(kernels), gpu_operation_count=len(ops),
        gpu_busy_union_ms=duration(iv(ops))/1e6,
        first_gpu_work_ms=(union[0][0]-lo)/1e6, after_last_gpu_work_ms=(hi-union[-1][1])/1e6,
        internal_gpu_idle_ms=sum(r['duration_ms'] for r in gaps),
        gaps_over_100us=sum(r['duration_ms']>=.1 for r in gaps),
        gaps_over_1ms=sum(r['duration_ms']>=1 for r in gaps),
        largest_gaps=gaps[:15], streams=sorted({r['streamId'] for r in ops}),
        sync_summary=sync_stats, cuda_graph_launch_count=len(graph_apis),
        nvtx_graph_replay_count=sum(r['stage']=='cuda_graph_replay' for r in nvtx),
        main_critical_path=main_critical, diagnostics=diagnostic,
    )
    assert abs(summary['gpu_busy_union_ms']+summary['first_gpu_work_ms']+
               summary['after_last_gpu_work_ms']+summary['internal_gpu_idle_ms']-
               summary['inference_wall_ms']) < 1e-5
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    plot(out, nvtx, ops, summary)
    print(json.dumps({k:v for k,v in summary.items() if k in [
        'inference_wall_ms','gpu_busy_union_ms','first_gpu_work_ms','after_last_gpu_work_ms',
        'internal_gpu_idle_ms','streams','cuda_graph_launch_count','overlap']},indent=2))


def plot(out, nvtx, ops, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    lo = summary['inference_start_ns']
    main_pid = summary['main_pid']
    colors = {'jpeg_read_decode':'#3479b8','cpu_preprocess':'#339e91','batch_assemble':'#805bad',
              'loader_wait':'#dc923a','h2d_submit':'#d7a429','gpu_forward_eager_submit':'#498b50',
              'gpu_postprocess_submit':'#93b360','d2h':'#755bb1','results_format':'#7e8793',
              'output_write':'#252d3c','Kernel':'#498b50','H2D':'#d7a429','D2H':'#755bb1',
              'D2D':'#5ba6aa','Memset':'#999999'}

    def render(name, start, end, title):
        workers=summary['worker_pids']
        lanes=[f'CPU worker {i+1} (PID {pid})' for i,pid in enumerate(workers)]+[
            'Main: wait / submit / result / write', 'GPU kernels (stream 7)', 'GPU memory copies / memset']
        fig, ax=plt.subplots(figsize=(17,7.2))
        rows=[]
        for i,pid in enumerate(workers):
            rows += [(i,r['stage'],r) for r in nvtx if r['pid']==pid and r['stage'] in ['jpeg_read_decode','cpu_preprocess','batch_assemble']]
        rows += [(len(workers),r['stage'],r) for r in nvtx if r['pid']==main_pid and r['stage'] in ['loader_wait','h2d_submit','gpu_forward_eager_submit','gpu_postprocess_submit','d2h','results_format','output_write']]
        rows += [(len(workers)+1 if r['kind']=='Kernel' else len(workers)+2,r['kind'],r) for r in ops]
        grouped=defaultdict(list)
        for lane,kind,r in rows:
            a=max(start,(r['start']-lo)/1e6); b=min(end,(r['end']-lo)/1e6)
            if b>a: grouped[lane,kind].append((a,b-a))
        for (lane,kind),bars in grouped.items():
            ax.broken_barh(bars,(lane-.33,.66),facecolors=colors[kind],linewidth=0)
        for r in nvtx:
            if r['stage']!='batch_consume_total': continue
            a=(r['start']-lo)/1e6
            if start<=a<=end:
                ax.axvline(a,color='#aab2bd',alpha=.35,lw=.7)
                if end-start<700:
                    ax.text(a,-.65,f'B{r["batch"]:02}',fontsize=9,rotation=45)
        ax.set(xlim=(start,end),ylim=(len(lanes)-.4,-1.0),yticks=range(len(lanes)),yticklabels=lanes,
               xlabel='Milliseconds since inference NVTX range start',title=title)
        ax.grid(axis='x',alpha=.2)
        legend=[Patch(facecolor=colors[k],label=k) for k in ['jpeg_read_decode','cpu_preprocess','batch_assemble','loader_wait','Kernel','H2D','D2H','results_format','output_write']]
        ax.legend(handles=legend,ncol=5,loc='upper center',bbox_to_anchor=(.5,-.12),fontsize=9)
        fig.text(.02,.015,'Source: actual Nsight SQLite start/end timestamps. Blank GPU lanes mean no traced work, not an occupancy estimate.',fontsize=10)
        fig.tight_layout(rect=(0,.08,1,1))
        fig.savefig(out/name,dpi=160)
        plt.close(fig)
    render('trace-overview.png',0,summary['inference_wall_ms'],'Full measured inference: CPU workers, main thread and GPU')
    b10=[r for r in nvtx if r['stage']=='batch_consume_total' and r['batch'] in [9,10,11]]
    render('trace-batches-09-11.png',max(0,min(r['start'] for r in b10)-lo)/1e6,
           (max(r['end'] for r in b10)-lo)/1e6,'Measured steady-state batches 09-11')
    gap=summary['largest_gaps'][0]
    render('trace-largest-gap.png',max(0,gap['start_ms']-15),min(summary['inference_wall_ms'],gap['start_ms']+gap['duration_ms']+15),
           f'Largest internal GPU work gap: {gap["duration_ms"]:.3f} ms; CPU stage {gap["dominant_cpu_stage"]}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sqlite',default='analysis/nsys/full_inference.sqlite')
    p.add_argument('--output',default='analysis/nsys')
    args=p.parse_args()
    analyze(Path(args.sqlite),Path(args.output))
