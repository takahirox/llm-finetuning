"""Execute MBPP assertions only inside a bounded, disposable Docker container."""

import json
import re
import subprocess
import uuid


# No bind mounts, host networking, credentials, or host Python execution.
RUNNER = """
import json, os, random, resource, sys
payload = json.load(sys.stdin)
resource.setrlimit(resource.RLIMIT_CPU, (payload['cpu_seconds'], payload['cpu_seconds']))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
random.seed(0)
with open(os.devnull, 'w') as sink:
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


def score(solution, problem, image, settings):
    name = 'mbpp-' + uuid.uuid4().hex
    command = [
        'docker', 'run', '--name', name, '--pull', 'never', '--rm', '-i',
        '--network', 'none', '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--user', '65534:65534',
        '--pids-limit', '32', '--memory', f"{settings['memory_mb']}m",
        '--memory-swap', f"{settings['memory_mb']}m", '--cpus', '1',
        '--ulimit', 'nofile=64:64', '--log-driver', 'none',
        '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m',
        '--env', 'PYTHONHASHSEED=0', image, 'python', '-s', '-B', '-c', RUNNER,
    ]
    payload = dict(solution=solution, setup=problem.get('test_setup_code', ''),
                   tests=problem['test_list'], cpu_seconds=settings['cpu_seconds'])
    try:
        result = subprocess.run(
            command, input=json.dumps(payload), text=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=settings['wall_seconds'],
        )
        if result.returncode in (125, 126, 127):
            raise ValueError(f'Docker execution failed (exit {result.returncode}); evaluation aborted.')
        return {'passed': result.returncode == 0,
                'status': 'passed' if result.returncode == 0 else 'failed',
                'exit_code': result.returncode}
    except subprocess.TimeoutExpired:
        return {'passed': False, 'status': 'timeout', 'exit_code': None}
    finally:
        # Killing the Docker CLI alone does not stop the container.
        cleanup_error = (f'Could not clean up evaluation container {name}; '
                         f'run docker rm -f {name} and confirm removal before continuing.')
        try:
            removal = subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, timeout=15, check=False)
            if removal.returncode != 0:
                # --rm may already have removed it. A failed inspect cannot
                # distinguish absence from a daemon error; a successful listing can.
                remaining = subprocess.run(
                    ['docker', 'container', 'ls', '--all', '--quiet', '--filter', f'name=^/{name}$'],
                    capture_output=True, text=True, timeout=15, check=False,
                )
                if remaining.returncode != 0 or remaining.stdout.strip():
                    raise ValueError(cleanup_error)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(cleanup_error) from exc
