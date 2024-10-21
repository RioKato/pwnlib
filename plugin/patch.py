from contextlib import contextmanager
from subprocess import Popen
from typing import IO, Iterator
import pwnlib


__popen = pwnlib.popen
__pclose = pwnlib.pclose


def pprint(message: str):
    from sys import stderr

    message = f'{pwnlib.Color.RED}[DEBUG]{pwnlib.Color.END} {message}'
    print(message, file=stderr, flush=True)


def popen(command: list[str], trace: bool,
          stdin: IO | int | None, stdout: IO | int | None, stderr: IO | int | None) -> Popen:
    from shlex import join

    p = __popen(command, trace, stdin, stdout, stderr)
    message = join(command)
    message = f'{message} (pid={p.pid})'
    pprint(message)
    return p


@contextmanager
def pclose(proc: Popen) -> Iterator[Popen]:
    from signal import Signals

    with __pclose(proc):
        yield proc

    code = proc.returncode

    if code >= 0:
        reason = f'pid={proc.pid}, code={code}'
    else:
        signal = Signals(-code).name
        reason = f'pid={proc.pid}, signal={signal}'

    message = f'process has terminated ({reason})'
    pprint(message)


pwnlib.popen = popen
pwnlib.pclose = pclose
