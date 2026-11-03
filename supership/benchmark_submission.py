"""Measure end-to-end inference after an untimed initialization stage."""

import argparse
import json
import math
import os
import selectors
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__:
    from .submission import validate_output
else:
    from submission import validate_output


READY_MARKER = "THU_RS_READY"


def parse_args():
    parser = argparse.ArgumentParser(
        usage=(
            "%(prog)s --image-dir DIR --meta-file FILE --output-file FILE "
            "[options] -- COMMAND [ARG ...]"
        )
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--meta-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--init-timeout", type=float, default=300.0)
    parser.add_argument("--run-timeout", type=float, default=3600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a worker command is required after --")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if any(not math.isfinite(value) or value <= 0
           for value in (args.init_timeout, args.run_timeout)):
        parser.error("timeouts must be finite and positive")
    return args


def wait_for_ready(process, timeout, completed=None):
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    pending = b""
    oversized_line = False

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"init did not finish within {timeout:.1f} seconds")
            if completed is not None and completed.is_set():
                raise RuntimeError("worker exited before READY")
            events = selector.select(min(remaining, 0.1))
            if not events:
                continue
            # TextIOWrapper can read READY ahead of select(), or block on a
            # partial line. Read only available bytes and frame lines here.
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                code = process.poll()
                raise RuntimeError(f"worker exited before READY (exit code {code})")
            pending += chunk
            lines = pending.split(b"\n")
            pending = lines.pop()
            for line in lines:
                if not oversized_line and line.strip() == READY_MARKER.encode("ascii"):
                    return
                oversized_line = False
            if len(pending) > 65536:
                pending = b""
                oversized_line = True
    finally:
        selector.close()


def output_path_for_run(base_path, run_index, runs):
    if runs == 1:
        return base_path
    path = Path(base_path)
    return str(path.with_name(f"{path.stem}.run{run_index + 1}{path.suffix}"))


def live_group_members(group_id):
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == group_id and fields[0] not in ("Z", "X"):
            members.append(int(entry.name))
    return members


def wait_for_group(group_id, deadline):
    waited = False
    while live_group_members(group_id):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("inference exceeded timeout with subprocesses still running")
        time.sleep(min(0.01, remaining))
        waited = True
    return time.monotonic_ns() if waited else None


def terminate(process, process_group=False):
    if process_group:
        # A worker can launch DataLoader or backend subprocesses. The session
        # belongs to this run, so cleanup must include those descendants.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 0.2
        while live_group_members(process.pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        wait_for_group(process.pid, time.monotonic() + 5)
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def drain_stdout(stream, stopped):
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        while not stopped.is_set():
            if selector.select(0.1) and not os.read(stream.fileno(), 65536):
                return


def record_exit(process, completed, result):
    result["return_code"] = process.wait()
    result["end_ns"] = time.monotonic_ns()
    completed.set()


def send_request(stream, request, deadline):
    payload = memoryview((request + "\n").encode("utf-8"))
    os.set_blocking(stream.fileno(), False)
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_WRITE)
        while payload:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("inference exceeded timeout while sending request")
            if not selector.select(remaining):
                continue
            try:
                payload = payload[os.write(stream.fileno(), payload):]
            except BlockingIOError:
                continue


def run_once(args, run_index):
    output_file = output_path_for_run(args.output_file, run_index, args.runs)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    process = subprocess.Popen(
        args.command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
        start_new_session=True,
    )
    completed = threading.Event()
    exit_result = {}
    waiter = threading.Thread(target=record_exit, args=(process, completed, exit_result))
    waiter.start()
    drainer = None
    stop_draining = threading.Event()

    try:
        wait_for_ready(process, args.init_timeout, completed)
        drainer = threading.Thread(
            target=drain_stdout, args=(process.stdout, stop_draining), daemon=True
        )
        drainer.start()
        request = json.dumps(
            {
                "image_dir": os.path.abspath(args.image_dir),
                "meta_file": os.path.abspath(args.meta_file),
                "output_file": os.path.abspath(output_file),
            },
            separators=(",", ":"),
        )

        start_ns = time.monotonic_ns()
        deadline = time.monotonic() + args.run_timeout
        send_request(process.stdin, request, deadline)
        process.stdin.close()
        if not completed.wait(timeout=max(0, deadline - time.monotonic())):
            raise TimeoutError(
                f"inference exceeded {args.run_timeout:.1f} seconds"
            )
        return_code = exit_result["return_code"]
        if return_code != 0:
            raise RuntimeError(f"worker failed with exit code {return_code}")
        group_end_ns = wait_for_group(process.pid, deadline)
        end_ns = group_end_ns if group_end_ns is not None else exit_result["end_ns"]
        elapsed_seconds = (end_ns - start_ns) / 1_000_000_000
        detection_count = validate_output(output_file, args.meta_file)
        return elapsed_seconds, output_file, detection_count
    finally:
        try:
            terminate(process, process_group=True)
        finally:
            stop_draining.set()
            waiter.join()
            if drainer is not None:
                drainer.join()
            process.stdout.close()
            process.stdin.close()


def main():
    args = parse_args()
    durations = []
    for run_index in range(args.runs):
        elapsed, output_file, detection_count = run_once(args, run_index)
        durations.append(elapsed)
        print(
            f"run {run_index + 1}/{args.runs}: {elapsed:.6f} s, "
            f"{detection_count} detections, {output_file}",
            flush=True,
        )

    report = {
        "runs_seconds": durations,
        "median_seconds": statistics.median(durations),
    }
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        raise SystemExit(1)
