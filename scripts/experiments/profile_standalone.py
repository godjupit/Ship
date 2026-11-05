"""Instrument the production standalone worker; timings are diagnostic only.

Launch with the same preset/environment as run.sh. PROFILE_JSON selects the
report path. CPU thread time excludes blocking, while summed wall times across
decoder threads overlap and must not be added to end-to-end latency.
"""
from collections import defaultdict
import functools
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from supership import trt_worker as worker
from cuda.bindings import driver

samples = defaultdict(list)
active = False
gpu_samples = []


def instrument(owner, name, label):
    original = getattr(owner, name)

    @functools.wraps(original)
    def measured(*args, **kwargs):
        if not active:
            return original(*args, **kwargs)
        wall, cpu = time.perf_counter(), time.thread_time()
        try:
            return original(*args, **kwargs)
        finally:
            samples[label].append((time.perf_counter() - wall, time.thread_time() - cpu))

    setattr(owner, name, measured)


def instrument_gpu(name):
    original = getattr(driver, name)

    def measured(*args, **kwargs):
        if not active:
            return original(*args, **kwargs)
        start = worker.check(driver.cuEventCreate(0))
        end = worker.check(driver.cuEventCreate(0))
        stream = args[-1]
        worker.check(driver.cuEventRecord(start, stream))
        result = original(*args, **kwargs)
        worker.check(driver.cuEventRecord(end, stream))
        gpu_samples.append((name, start, end))
        return result

    setattr(driver, name, measured)


original_run = worker.Runtime.run


def profiled_run(runtime, request):
    global active
    active = True
    started = time.perf_counter()
    try:
        return original_run(runtime, request)
    finally:
        elapsed = time.perf_counter() - started
        active = False
        report = dict(run_wall_seconds=elapsed, stages={
            name: dict(calls=len(values), wall_sum_seconds=sum(v[0] for v in values),
                       thread_cpu_sum_seconds=sum(v[1] for v in values))
            for name, values in samples.items()
        })
        if gpu_samples:
            anchor = gpu_samples[0][1]
            report['gpu_timeline_ms'] = [dict(
                operation=name,
                start=worker.check(driver.cuEventElapsedTime(anchor, start)),
                end=worker.check(driver.cuEventElapsedTime(anchor, end)),
            ) for name, start, end in gpu_samples]
            totals = defaultdict(float)
            for item in report['gpu_timeline_ms']:
                totals[item['operation']] += item['end'] - item['start']
            report['gpu_operation_ms'] = dict(totals)
            for _, start, end in gpu_samples:
                worker.check(driver.cuEventDestroy(start))
                worker.check(driver.cuEventDestroy(end))
        path = Path(os.environ.get('PROFILE_JSON', 'analysis/standalone-stages.json'))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'gpu_timeline_ms'}),
              file=sys.stderr, flush=True)


if __name__ == '__main__':
    instrument(worker.cv2, 'imread', 'read_decode')
    if os.environ.get('STANDALONE_DECODER', 'opencv') != 'opencv':
        from supership.jpeg_decoder import JPEGDecoder
        instrument(JPEGDecoder, '__call__', 'jpeg_read_decode')
    instrument(worker.cv2, 'resize', 'resize')
    instrument(worker, 'serialize', 'serialize')
    instrument(os, 'fsync', 'fsync')
    for name in ('cuMemcpyHtoDAsync', 'cuMemcpyDtoHAsync', 'cuGraphLaunch',
                 'cuEventSynchronize', 'cuStreamSynchronize'):
        instrument(driver, name, name)
    if os.environ.get('PROFILE_CUDA_EVENTS', '0') == '1':
        for name in ('cuMemcpyHtoDAsync', 'cuMemcpyDtoHAsync', 'cuGraphLaunch'):
            instrument_gpu(name)
    worker.Runtime.run = profiled_run
    worker.main()
