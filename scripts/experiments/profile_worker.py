"""Profile only the standalone worker's request and flush CUPTI before exit."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from supership import trt_worker

original_run = trt_worker.Runtime.run


def profiled_run(runtime, request):
    # Model loading, graph construction and synthetic warmup have completed.
    # An explicit stop flushes the capture before main uses os._exit().
    trt_worker.check(runtime.cu.cuProfilerStart())
    try:
        return original_run(runtime, request)
    finally:
        trt_worker.check(runtime.cu.cuProfilerStop())


if __name__ == '__main__':
    trt_worker.Runtime.run = profiled_run
    trt_worker.main()
