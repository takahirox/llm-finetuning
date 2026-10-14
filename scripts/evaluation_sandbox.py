"""Execute MBPP assertions only inside a bounded, disposable Docker container."""

import json
import os
import re
import select
import subprocess
import time
import uuid


READY = b'MBPP_READY\n'
# No bind mounts, host networking, credentials, or host Python execution.
# Readiness follows payload delivery/setup and precedes any generated-code execution.
RUNNER = """
import json, os, random, resource, signal, sys
payload = json.loads(sys.stdin.readline())
resource.setrlimit(resource.RLIMIT_CPU, (payload['cpu_seconds'], payload['cpu_seconds']))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
random.seed(0)
# An independent in-container wall timer excludes Docker CLI exit-delivery delays.
signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
print('MBPP_READY', flush=True)
signal.setitimer(signal.ITIMER_REAL, payload['wall_seconds'])
with open(os.devnull, 'w') as sink:
    os.dup2(sink.fileno(), 1)
    os.dup2(sink.fileno(), 2)
    sys.stdout = sink
    sys.stderr = sink
    scope = {'__name__': '__main__'}
    try:
        exec(compile(payload['setup'], '<setup>', 'exec'), scope)
        exec(compile(payload['solution'], '<solution>', 'exec'), scope)
        for test in payload['tests']:
            exec(compile(test, '<test>', 'exec'), scope)
    except BaseException:
        sys.exit(1)
"""


def resolve_image(image):
    """Resolve an already pulled image to an immutable local image ID."""
    try:
        result = subprocess.run(
            ['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
            capture_output=True, text=True, check=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('Docker/image unavailable. Start Docker and pull the documented Python image.') from exc
    image_id = result.stdout.strip()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
        raise ValueError('Docker returned an invalid image ID.')
    return image_id


def await_ready(process, seconds, payload):
    """Bound container startup, payload delivery, and trusted Python setup together."""
    deadline = time.monotonic() + seconds
    received = b''
    pending = memoryview(payload + b'\n')
    os.set_blocking(process.stdin.fileno(), False)
    while received != READY:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError('Docker startup/readiness timeout; evaluation aborted (not a model failure).')
        readable, writable, _ = select.select([process.stdout], [process.stdin] if pending else [], [], remaining)
        if not readable and not writable:
            raise ValueError('Docker startup/readiness timeout; evaluation aborted (not a model failure).')
        if writable:
            try:
                sent = os.write(process.stdin.fileno(), pending[:4096])
                pending = pending[sent:]
            except BlockingIOError:
                pass
        if readable:
            block = os.read(process.stdout.fileno(), len(READY) - len(received))
            received += block
            if not block or not READY.startswith(received):
                raise ValueError('Docker failed before readiness; evaluation aborted.')
    if pending:
        raise ValueError('Docker readiness arrived before payload delivery; evaluation aborted.')


def container_state(name):
    """A CLI exit alone cannot distinguish model failure from a daemon failure."""
    try:
        result = subprocess.run(
            ['docker', 'inspect', '--format', '{{json .State}}', name],
            capture_output=True, text=True, timeout=15, check=True,
        )
        state = json.loads(result.stdout)
        if not isinstance(state, dict) or state.get('Error') or state.get('Status') not in ('running', 'exited'):
            raise ValueError('Invalid container state.')
        if state['Status'] == 'exited' and type(state.get('ExitCode')) is not int:
            raise ValueError('Invalid container exit code.')
        return state
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError('Docker state unavailable or failed; evaluation aborted.') from exc


def remove_container(name):
    cleanup_error = (f'Could not clean up evaluation container {name}; '
                     f'run docker rm -f {name} and confirm removal before continuing.')
    try:
        removal = subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, timeout=15, check=False)
        if removal.returncode != 0:
            remaining = subprocess.run(
                ['docker', 'container', 'ls', '--all', '--quiet', '--filter', f'name=^/{name}$'],
                capture_output=True, text=True, timeout=15, check=False,
            )
            if remaining.returncode != 0 or remaining.stdout.strip():
                raise ValueError(cleanup_error)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(cleanup_error) from exc


def exit_outcome(code):
    if type(code) is not int or code in (125, 126, 127):
        raise ValueError(f'Docker execution failed (exit {code}); evaluation aborted.')
    if code == 124:
        return {'passed': False, 'status': 'timeout', 'exit_code': None}
    return {'passed': code == 0, 'status': 'passed' if code == 0 else 'failed', 'exit_code': code}


def score(solution, problem, image, settings):
    name = 'mbpp-' + uuid.uuid4().hex
    command = [
        'docker', 'run', '--name', name, '--pull', 'never', '-i',
        '--network', 'none', '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--user', '65534:65534',
        '--pids-limit', '32', '--memory', f"{settings['memory_mb']}m",
        '--memory-swap', f"{settings['memory_mb']}m", '--cpus', '1',
        '--ulimit', 'nofile=64:64', '--log-driver', 'none',
        '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m',
        '--env', 'PYTHONHASHSEED=0', image, 'python', '-s', '-B', '-c', RUNNER,
    ]
    payload = json.dumps(dict(solution=solution, setup=problem.get('test_setup_code', ''),
                              tests=problem['test_list'], cpu_seconds=settings['cpu_seconds'],
                              wall_seconds=settings['wall_seconds'])).encode()
    process = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, bufsize=0)
        await_ready(process, settings.get('startup_seconds', 30), payload)
        ready = time.monotonic()
        try:
            # The wall deadline starts only after trusted Python is ready.
            process.stdin.close()
            process.stdin = None
            process.communicate(timeout=settings['wall_seconds'])
        except subprocess.TimeoutExpired:
            executed = time.monotonic() - ready
            state = container_state(name)  # Fail closed if Docker is unavailable.
            # Docker can deliver the CLI exit late. The runner's own wall timer
            # guarantees that a completed program did not exceed its deadline.
            outcome = exit_outcome(state['ExitCode']) if state['Status'] == 'exited' else {
                'passed': False, 'status': 'timeout', 'exit_code': None}
        else:
            executed = time.monotonic() - ready
            state = container_state(name)
            if state['Status'] != 'exited' or state.get('ExitCode') != process.returncode:
                raise ValueError('Docker exit/state mismatch; evaluation aborted.')
            outcome = exit_outcome(process.returncode)
        return {**outcome, 'startup_seconds': ready - started, 'execution_wait_seconds': executed}
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('Docker execution unavailable; evaluation aborted.') from exc
    finally:
        try:
            if process is not None and process.poll() is None:
                process.kill()
            if process is not None:
                process.wait(timeout=15)
        finally:
            try:
                remove_container(name)
            finally:
                if process is not None:
                    for stream in (process.stdin, process.stdout):
                        if stream is not None:
                            stream.close()
