"""Check allocator settings reach either worker before its interpreter starts."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not os.sys.platform.startswith('linux'), reason='Linux allocator setting')
@pytest.mark.parametrize('pipeline', ['standard', 'fast'])
@pytest.mark.parametrize('host_thp,tunables,expected', [
    (None, None, 'glibc.malloc.hugetlb=1'),
    ('0', None, None),
    ('1', 'glibc.malloc.arena_max=2', 'glibc.malloc.arena_max=2:glibc.malloc.hugetlb=1'),
    ('1', 'glibc.malloc.hugetlb=0:glibc.malloc.arena_max=2',
     'glibc.malloc.hugetlb=0:glibc.malloc.arena_max=2'),
])
def test_allocator_environment_at_worker_start(tmp_path, pipeline, host_thp, tunables, expected):
    executable = tmp_path / 'inspect-worker'
    executable.write_text('#!/usr/bin/env python3\nimport os,json,sys\n'
                          'print(json.dumps([os.environ.get("GLIBC_TUNABLES"),sys.argv[1]]))\n')
    executable.chmod(0o755)
    env = os.environ.copy()
    for key in ['HOST_THP', 'GLIBC_TUNABLES', 'BACKEND', 'TRT_STANDALONE']:
        env.pop(key, None)
    env.update(PYTHON_BIN=str(executable), PIPELINE=pipeline)
    if host_thp is not None:
        env['HOST_THP'] = host_thp
    if tunables is not None:
        env['GLIBC_TUNABLES'] = tunables
    result = subprocess.run(['bash', str(ROOT / 'run.sh'), '--worker'], env=env,
                            capture_output=True, text=True, check=True)
    actual, worker = json.loads(result.stdout)
    assert actual == expected
    assert worker == ('supership/trt_worker.py' if pipeline == 'fast'
                      else 'supership/generate_submission.py')


@pytest.mark.parametrize('entrypoint', ['run.sh', 'tools/benchmark/benchmark_optimized.sh'])
@pytest.mark.parametrize('overrides', [{}, {'INPUT_SIZE': '640', 'THREADED_JSON': '0', 'NUM_WORKERS': '3'}])
@pytest.mark.parametrize('allocated_cpus,workers', [(None, '9'), ('4', '9'), ('8', '6')])
def test_default_launcher_selects_verified_fastest_and_preserves_overrides(tmp_path, overrides, entrypoint, allocated_cpus, workers):
    executable = tmp_path / 'inspect-preset'
    executable.write_text('#!/usr/bin/env python3\nimport os,json,sys\n'
                          'keys=["PIPELINE","INPUT_SIZE","NUM_WORKERS","THREADED_JSON","POST_TOP_QUERIES","INFERENCE_IMPORTS"]\n'
                          'print(json.dumps([sys.argv[1],{k:os.environ.get(k) for k in keys}]))\n')
    executable.chmod(0o755)
    env = os.environ.copy()
    keys = ['PIPELINE', 'INPUT_SIZE', 'NUM_WORKERS', 'THREADED_JSON', 'POST_TOP_QUERIES', 'INFERENCE_IMPORTS']
    for key in keys:
        env.pop(key, None)
    for key in ['BACKEND', 'TRT_STANDALONE', 'TRT_ENGINE']:
        env.pop(key, None)
    env.pop('SLURM_CPUS_PER_TASK', None)
    if allocated_cpus is not None:
        env['SLURM_CPUS_PER_TASK'] = allocated_cpus
    env.update(PYTHON_BIN=str(executable), **overrides)
    result = subprocess.run(['bash', str(ROOT / entrypoint), '--worker'], env=env,
                            capture_output=True, text=True, check=True)
    worker, config = json.loads(result.stdout)
    assert worker == ('supership/trt_worker.py' if entrypoint == 'run.sh'
                      else 'supership/benchmark_submission.py')
    assert config == dict(PIPELINE='fast', INPUT_SIZE='544', NUM_WORKERS=workers,
                          THREADED_JSON='1', POST_TOP_QUERIES='100', INFERENCE_IMPORTS='1') | overrides
