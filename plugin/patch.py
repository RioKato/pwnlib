from contextlib import contextmanager
from subprocess import Popen
from typing import Iterator
import pwnlib


__pclose = pwnlib.pclose


@contextmanager
def pclose(popen: Popen) -> Iterator[Popen]:
    from sys import stderr
    from signal import Signals

    with __pclose(popen):
        yield popen

    code = popen.returncode

    if code >= 0:
        reason = f'pid={popen.pid}, code={code}'
    else:
        signal = Signals(-code).name
        reason = f'pid={popen.pid}, signal={signal}'

    message = f'{pwnlib.Color.RED}[DEBUG]{pwnlib.Color.END} process has terminated({reason})'
    print(message, file=stderr)


pwnlib.pclose = pclose
