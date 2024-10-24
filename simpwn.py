from abc import ABCMeta, abstractmethod
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
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


class Runner(Command):
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
    def debug(self) -> list[str]:
        ...

    @contextmanager
    def launch(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[Callable[[], None]]:
        with popen(self.prepare(env=env, aslr=aslr), False, redirect, False):
            with popen(self.debug(), False, redirect, True):
                yield lambda: None


class Attacher(Runner):
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


class MultiAttacher(Runner):
    @abstractmethod
    def prepare(self) -> list[str]:
        ...

    @abstractmethod
    def attach(self, pid: int) -> list[str]:
        ...

    @contextmanager
    def launch(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[Callable[[], None]]:
        with popen(self.prepare(), False, None, False):
            with popen(self.run(env=env, aslr=aslr), True, redirect, False) as pid:
                with ExitStack() as estack:
                    def attach():
                        estack.enter_context(popen(self.attach(pid), False, None, True))

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


@dataclass
class _Target:
    command: list[str]
    env: str = 'env'
    setarch: str = 'setarch'

    def run(self, *, env: dict[str, str] = {}, aslr: bool = True) -> list[str]:
        command = []

        if not aslr:
            command += [self.setarch, '-R']

        if env:
            command += [self.env]

            for k, v in env.items():
                command += [f'{k}={v}']

        command += self.command
        return command


class Target(_Target, Runner):
    pass


@dataclass
class _Gdb(_Target):
    host: str = ''
    port: int = 1234
    file: str = ''
    sysroot: str = ''
    startup: str = 'target remote {host}:{port}'
    script: str = ''
    gdb: str = 'gdb'

    def open(self) -> list[str]:
        command = [self.gdb]

        if self.sysroot:
            command += ['-ex', f'set sysroot {self.sysroot}']

        startup = self.startup.format(host=self.host, port=self.port)
        command += ['-ex', startup]

        if self.script:
            command += ['-x', self.script]

        if self.file:
            command += [self.file]

        return command


class Gdb:
    @dataclass
    class Debugger(_Gdb, Debugger):
        options: list[str] = field(default_factory=list)
        gdbserver: str = 'gdbserver'
        env: str = 'env'

        def debug(self, *, env: dict[str, str] = {}, aslr: bool = True) -> list[str]:
            command = [self.gdbserver]

            if aslr:
                command += ['--no-disable-randomization']
            else:
                command += ['--disable-randomization']

            if env:
                command += ['--wrapper', self.env]

                for k, v in env.items():
                    command += [f'{k}={v}']

                command += ['--']

            command += self.options
            command += [f'{self.host}:{self.port}']
            command += self.run(env={}, aslr=True)
            return command

    @dataclass
    class Attacher(_Gdb, Attacher):
        gdbserver: str = 'gdbserver'

        def attach(self, pid: int) -> list[str]:
            return [self.gdbserver, '--attach', f'{self.host}:{self.port}', f'{pid}']


@dataclass
class RR(_Gdb, ReverseDebugger):
    options: list[str] = field(default_factory=list)
    rr: str = 'rr'
    setarch: str = 'setarch'

    def record(self, *, env: dict[str, str] = {}, aslr: bool = True) -> list[str]:
        command = []

        if not aslr:
            command += [self.setarch, '-R']

        command += [self.rr, 'record']

        for k, v in env.items():
            command += ['-v', f'{k}={v}']

        command += self.options
        command += self.run(env={}, aslr=True)
        return command

    def replay(self) -> list[str]:
        return [self.rr, 'replay', '-s', f'{self.port}']


class Tmux:
    @dataclass
    class Debugger(Debugger):
        command: Debugger
        options: list[str] = field(default_factory=list)
        tmux: str = 'tmux'

        def debug(self, *, env: dict[str, str] = {}, aslr: bool = True) -> list[str]:
            return self.command.debug(env=env, aslr=aslr)

        def open(self) -> list[str]:
            command = [self.tmux, 'split']
            command += self.options
            command += self.command.open()
            return command

    @dataclass
    class MultiDebugger(MultiDebugger):
        command: MultiDebugger
        options: list[str] = field(default_factory=list)
        tmux: str = 'tmux'

        def prepare(self, *, env: dict[str, str] = {}, aslr: bool = True) -> list[str]:
            return self.command.prepare(env=env, aslr=aslr)

        def debug(self) -> list[str]:
            command = [self.tmux, 'split']
            command += self.options
            command += self.command.debug()
            return command

    @dataclass
    class Attacher(Attacher):
        command: Attacher
        options: list[str] = field(default_factory=list)
        tmux: str = 'tmux'

        def run(self, *, env: dict[str, str] = {}, aslr: bool = True) -> list[str]:
            return self.command.run(env=env, aslr=aslr)

        def attach(self, pid: int) -> list[str]:
            return self.command.attach(pid)

        def open(self) -> list[str]:
            command = [self.tmux, 'split']
            command += self.options
            command += self.command.open()
            return command

    @dataclass
    class MultiAttacher(MultiAttacher):
        command: MultiAttacher
        options: list[str] = field(default_factory=list)
        tmux: str = 'tmux'

        def run(self, *, env: dict[str, str] = {}, aslr: bool = True) -> list[str]:
            return self.command.run(env=env, aslr=aslr)

        def prepare(self) -> list[str]:
            return self.command.prepare()

        def attach(self, pid: int) -> list[str]:
            command = [self.tmux, 'split']
            command += self.options
            command += self.command.attach(pid)
            return command

    @dataclass
    class ReverseDebugger(ReverseDebugger):
        command: ReverseDebugger
        options: list[str] = field(default_factory=list)
        tmux: str = 'tmux'

        def record(self, *, env: dict[str, str] = {}, aslr: bool = True) -> list[str]:
            return self.command.record(env=env, aslr=aslr)

        def replay(self) -> list[str]:
            return self.command.replay()

        def open(self) -> list[str]:
            command = [self.tmux, 'split']
            command += self.options
            command += self.command.open()
            return command
