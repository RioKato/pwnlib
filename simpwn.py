from abc import ABCMeta, abstractmethod
from contextlib import ExitStack, contextmanager, suppress
from socket import socket
from typing import IO, Callable, Iterator, cast


class Glibc:
    from ctypes import CDLL, c_int, c_ulong

    PR_SET_PTRACER = 0x59616d61
    PR_SET_PTRACER_ANY = -1

    prctl = CDLL(None).prctl
    prctl.restype = c_int
    prctl.argtypes = [c_int, c_ulong, c_ulong, c_ulong, c_ulong]


@contextmanager
def popen(command: list[str], trace: bool, redirect: socket | None, interactive: bool) -> Iterator[int]:
    from subprocess import DEVNULL, Popen, TimeoutExpired

    def pr_set_ptracer_any():
        Glibc.prctl(Glibc.PR_SET_PTRACER, Glibc.PR_SET_PTRACER_ANY, 0, 0, 0)

    if redirect:
        stdin = stdout = stderr = cast(IO, redirect)
    else:
        stdin = None if interactive else DEVNULL
        stdout = stderr = None

    preexec_fn = pr_set_ptracer_any if trace else None

    with Popen(command, stdin=stdin, stdout=stdout, stderr=stderr, process_group=0, preexec_fn=preexec_fn) as proc:
        try:
            yield proc.pid
        finally:
            if proc.poll() is None:
                proc.terminate()

                try:
                    proc.wait(1)
                except TimeoutExpired:
                    proc.kill()


class Command(metaclass=ABCMeta):
    @abstractmethod
    @contextmanager
    def launch(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[Callable[[], None]]:
        ...


class Target(Command):
    @abstractmethod
    def run(self, *, env: dict[str, str] = {}, aslr: bool = True) -> list[str]:
        ...

    @contextmanager
    def launch(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[Callable[[], None]]:
        with popen(self.run(env=env, aslr=aslr), False, redirect, False):
            yield lambda: None


class Debugger(Command):
    @abstractmethod
    def debug(self, *, env: dict[str, str] = {}, aslr: bool = True) -> list[str]:
        ...

    @abstractmethod
    def open(self) -> list[str]:
        ...

    @contextmanager
    def launch(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[Callable[[], None]]:
        with popen(self.debug(env=env, aslr=aslr), False, redirect, False):
            with popen(self.open(), False, redirect, True):
                yield lambda: None


class MultiDebugger(Command):
    @abstractmethod
    def prepare(self, *, env: dict[str, str] = {}, aslr: bool = True) -> list[str]:
        ...

    @abstractmethod
    def open(self) -> list[str]:
        ...

    @contextmanager
    def launch(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[Callable[[], None]]:
        with popen(self.prepare(env=env, aslr=aslr), False, redirect, False):
            with popen(self.open(), False, redirect, True):
                yield lambda: None


class Attacher(Target):
    @abstractmethod
    def attach(self, pid: int) -> list[str]:
        ...

    @abstractmethod
    def open(self) -> list[str]:
        ...

    @contextmanager
    def launch(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[Callable[[], None]]:
        with popen(self.run(env=env, aslr=aslr), True, redirect, False) as pid:
            with ExitStack() as estack:
                def attach():
                    estack.enter_context(popen(self.attach(pid), False, None, False))
                    estack.enter_context(popen(self.open(), False, None, True))

                yield attach


class MultiAttacher(Target):
    @abstractmethod
    def prepare(self) -> list[str]:
        ...

    @abstractmethod
    def open(self, pid: int) -> list[str]:
        ...

    @contextmanager
    def launch(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[Callable[[], None]]:
        with popen(self.prepare(), False, None, False):
            with popen(self.run(env=env, aslr=aslr), True, redirect, False) as pid:
                with ExitStack() as estack:
                    def attach():
                        estack.enter_context(popen(self.open(pid), False, None, True))

                    yield attach


class StopRecording(Exception):
    pass


class ReverseDebugger(Command):
    @abstractmethod
    def record(self, *, env: dict[str, str] = {}, aslr: bool = True) -> list[str]:
        ...

    @abstractmethod
    def replay(self) -> list[str]:
        ...

    @abstractmethod
    def open(self) -> list[str]:
        ...

    @contextmanager
    def launch(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[Callable[[], None]]:
        from signal import SIGINT, sigwait

        try:
            with popen(self.record(env=env, aslr=aslr), False, redirect, False):
                with suppress(StopRecording):
                    def replay():
                        raise StopRecording

                    yield replay
        finally:
            with popen(self.replay(), False, None, False):
                with popen(self.open(), False, None, True):
                    sigwait([SIGINT])
