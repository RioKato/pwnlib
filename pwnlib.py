from typing import IO, Callable, Iterator, Literal, Self, cast
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from threading import Thread

__all__ = [
    'Alias', 'Target', 'GdbServer', 'RR', 'Qemu', 'Docker', 'Tmux', 'Display',
    'Proxy', 'Context',
    'p8', 'p16', 'p32', 'p64', 'pf', 'pd',
    'u8', 'u16', 'u32', 'u64', 'uf', 'ud',
    'block',
    'rol64', 'ror64',
    'iota'
]


class Color:
    @staticmethod
    def __ansi(color: str) -> str:
        from sys import stdout

        return color if stdout.isatty() else ''

    END: str = __ansi('\033[0m')
    BOLD: str = __ansi('\033[1m')
    UNDERLINE: str = __ansi('\033[4m')
    REVERCE: str = __ansi('\033[07m')
    INVISIBLE: str = __ansi('\033[08m')
    BLACK: str = __ansi('\033[30m')
    RED: str = __ansi('\033[31m')
    GREEN: str = __ansi('\033[32m')
    YELLOW: str = __ansi('\033[33m')
    BLUE: str = __ansi('\033[34m')
    PURPLE: str = __ansi('\033[35m')
    CYAN: str = __ansi('\033[36m')
    WHITE: str = __ansi('\033[37m')


@dataclass
class Alias:
    sh: str = 'sh'
    env: str = 'env'
    setarch: str = 'setarch'
    gdb: str = 'gdb'
    gdbserver: str = 'gdbserver'
    rr: str = 'rr'
    docker: str = 'docker'
    tmux: str = 'tmux'


class Command(metaclass=ABCMeta):
    @property
    @abstractmethod
    def alias(self) -> Alias:
        pass

    @property
    @abstractmethod
    def executable(self) -> str:
        pass

    @abstractmethod
    def lookup[T: Command](self, cls: type[T]) -> T | None:
        pass

    def run(self, env: dict[str, str], aslr: bool) -> list[str]:
        return []

    def debug(self, env: dict[str, str], aslr: bool) -> list[str]:
        return []

    def attach(self, pid: int) -> list[str]:
        return []

    def record(self, env: dict[str, str], aslr: bool) -> list[str]:
        return []

    def replay(self) -> list[str]:
        return []

    def view(self) -> list[str]:
        return []

    @abstractmethod
    def getpid(self, pid: int) -> int:
        pass


@dataclass
class Target(Command):
    command: list[str]

    def __init__(self, command: list[str], alias: Alias | None = None):
        self.command = command
        self.__alias: Alias = alias if alias else Alias()

    @property
    def alias(self) -> Alias:
        return self.__alias

    @property
    def executable(self) -> str:
        return self.command[0] if self.command else ''

    def lookup[T: Command](self, cls: type[T]) -> T | None:
        if isinstance(self, cls):
            return self

        return None

    def run(self, env: dict[str, str], aslr: bool) -> list[str]:
        command = []

        if not aslr:
            command += [self.alias.setarch, '-R']

        if env:
            command += [self.alias.env]

            for k, v in env.items():
                command += [f'{k}={v}']

        command += self.command
        return command

    def getpid(self, pid: int) -> int:
        return pid


@dataclass
class Outer(Command):
    command: Command

    @property
    def alias(self) -> Alias:
        return self.command.alias

    @property
    def executable(self) -> str:
        return self.command.executable

    def lookup[T: Command](self, cls: type[T]) -> T | None:
        if isinstance(self, cls):
            return self

        return self.command.lookup(cls)

    def run(self, env: dict[str, str], aslr: bool) -> list[str]:
        return self.command.run(env, aslr)

    def debug(self, env: dict[str, str], aslr: bool) -> list[str]:
        return self.command.debug(env, aslr)

    def attach(self, pid: int) -> list[str]:
        return self.command.attach(pid)

    def record(self, env: dict[str, str], aslr: bool) -> list[str]:
        return self.command.record(env, aslr)

    def replay(self) -> list[str]:
        return self.command.replay()

    def view(self) -> list[str]:
        return self.command.view()

    def getpid(self, pid: int) -> int:
        return self.command.getpid(pid)


@dataclass
class Gdb(Outer):
    host: str = ''
    port: int = 1234
    file: str = ''
    sysroot: str = ''
    startup: str = 'target remote {host}:{port}'
    script: str = ''

    def view(self) -> list[str]:
        command = [self.alias.gdb]

        if self.sysroot:
            command += ['-ex', f'set sysroot {self.sysroot}']

        startup = self.startup.format(host=self.host, port=self.port)
        command += ['-ex', startup]

        if self.script:
            command += ['-x', self.script]

        if self.file:
            command += [self.file]

        return command


@dataclass
class GdbServer(Gdb):
    options: list[str] = field(default_factory=list)

    def debug(self, env: dict[str, str], aslr: bool) -> list[str]:
        command = [self.alias.gdbserver]

        if aslr:
            command += ['--no-disable-randomization']
        else:
            command += ['--disable-randomization']

        if env:
            command += ['--wrapper', self.alias.env]

            for k, v in env.items():
                command += [f'{k}={v}']

            command += ['--']

        command += self.options
        command += [f'{self.host}:{self.port}']
        command += self.command.run({}, True)
        return command

    def attach(self, pid: int) -> list[str]:
        return [self.alias.gdbserver, '--attach', f'{self.host}:{self.port}', f'{pid}']


@dataclass
class RR(Gdb):
    options: list[str] = field(default_factory=list)

    def record(self, env: dict[str, str], aslr: bool) -> list[str]:
        command = []

        if not aslr:
            command += [self.alias.setarch, '-R']

        command += [self.alias.rr, 'record']

        for k, v in env.items():
            command += ['-v', f'{k}={v}']

        command += self.options
        command += self.command.run({}, True)
        return command

    def replay(self) -> list[str]:
        return [self.alias.rr, 'replay', '-s', f'{self.port}']


@dataclass
class Qemu(Gdb):
    def __post_init__(self):
        self.file = self.file if self.file else self.executable

    def __qemu(self, env: dict[str, str], aslr: bool, debug: bool) -> list[str]:
        assert (not aslr)
        qemu_set_env = [f'{k}={v}' for (k, v) in env.items()]
        qemu_set_env = ','.join(qemu_set_env)
        env = {}

        if qemu_set_env:
            env['QEMU_SET_ENV'] = qemu_set_env

        if self.sysroot:
            env['QEMU_LD_PREFIX'] = self.sysroot

        if debug:
            env['QEMU_GDB'] = f'{self.port}'

        return self.command.run(env, True)

    def run(self, env: dict[str, str], aslr: bool) -> list[str]:
        return self.__qemu(env, aslr, False)

    def debug(self, env: dict[str, str], aslr: bool) -> list[str]:
        return self.__qemu(env, aslr, True)


@dataclass
class Docker(Outer):
    image: str
    name: str
    net: str = 'host'
    initcpid: int = 7
    options: list[str] = field(default_factory=list)

    def __docker_run(self, inner: list[str], *options: str) -> list[str]:
        command = [self.alias.docker, 'run', '-i', '--privileged', f'--name={self.name}']

        if self.net:
            command += [f'--net={self.net}']

        command += options
        command += self.options
        command += [self.image]
        command += inner
        return command

    def __docker_exec(self, inner: list[str]) -> list[str]:
        command = [self.alias.docker, 'exec', '-i', self.name]
        command += inner
        return command

    def __docker_start(self) -> list[str]:
        return [self.alias.docker, 'start', '-i', self.name]

    def run(self, env: dict[str, str], aslr: bool) -> list[str]:
        command = self.command.run(env, aslr)
        return self.__docker_run(command, '--init')

    def debug(self, env: dict[str, str], aslr: bool) -> list[str]:
        command = self.command.debug(env, aslr)
        return self.__docker_run(command)

    def attach(self, pid: int) -> list[str]:
        command = self.command.attach(pid)
        return self.__docker_exec(command)

    def record(self, env: dict[str, str], aslr: bool) -> list[str]:
        from shlex import join

        record = self.command.record(env, aslr)
        record = join(record)
        replay = self.command.replay()
        replay = join(replay)
        command = [self.alias.sh, '-c', f'{replay} > /dev/null 2>&1 || {record}']
        return self.__docker_run(command)

    def replay(self) -> list[str]:
        return self.__docker_start()

    def getpid(self, pid: int) -> int:
        # If Gdb attaches to a process with pid 1, interrupt command (C-c) will not work.
        # https://sourceware.org/bugzilla/show_bug.cgi?id=18945
        # The pid of the child process of the "--init" process inside the container is empirically 7.
        return self.initcpid


@dataclass
class Tmux(Outer):
    options: list[str] = field(default_factory=list)

    def view(self) -> list[str]:
        command = [self.alias.tmux, 'split']
        command += self.options
        command += self.command.view()
        return command


@dataclass
class Display(Outer):
    @staticmethod
    def __display(command: list[str]):
        from sys import stderr
        from shlex import join

        message = join(command)
        message = f'{Color.RED}[DEBUG]{Color.END} {message}'
        print(message, file=stderr, flush=True)

    def run(self, env: dict[str, str], aslr: bool) -> list[str]:
        command = self.command.run(env, aslr)
        self.__display(command)
        return command

    def debug(self, env: dict[str, str], aslr: bool) -> list[str]:
        command = self.command.debug(env, aslr)
        self.__display(command)
        return command

    def attach(self, pid: int) -> list[str]:
        command = self.command.attach(pid)
        self.__display(command)
        return command

    def record(self, env: dict[str, str], aslr: bool) -> list[str]:
        command = self.command.record(env, aslr)
        self.__display(command)
        return command

    def replay(self) -> list[str]:
        command = self.command.replay()
        self.__display(command)
        return command

    def view(self) -> list[str]:
        command = self.command.view()
        self.__display(command)
        return command


class Glibc:
    from ctypes import CDLL, c_int, c_ulong

    PR_SET_PTRACER = 0x59616d61
    PR_SET_PTRACER_ANY = -1

    prctl = CDLL(None).prctl
    prctl.restype = c_int
    prctl.argtypes = [c_int, c_ulong, c_ulong, c_ulong, c_ulong]


@dataclass
class Executor:
    from subprocess import Popen
    from socket import socket

    command: Command

    def __popen(self, command: list[str], redirect: socket | None, trace: bool) -> Popen:
        from subprocess import Popen, DEVNULL

        def pr_set_ptracer_any():
            Glibc.prctl(Glibc.PR_SET_PTRACER, Glibc.PR_SET_PTRACER_ANY, 0, 0, 0)

        stdin = stdout = stderr = cast(IO, redirect) if redirect else DEVNULL
        start_new_session = trace
        preexec_fn = pr_set_ptracer_any if trace else None
        return Popen(command, stdin=stdin, stdout=stdout, stderr=stderr,
                     start_new_session=start_new_session, preexec_fn=preexec_fn)

    def run(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Popen:
        command = self.command.run(env, aslr)
        return self.__popen(command, redirect, True)

    def debug(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Popen:
        command = self.command.debug(env, aslr)
        return self.__popen(command, redirect, False)

    def attach(self, pid: int) -> Popen:
        command = self.command.attach(pid)
        return self.__popen(command, None, False)

    def record(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Popen:
        command = self.command.record(env, aslr)
        return self.__popen(command, redirect, False)

    def replay(self) -> Popen:
        command = self.command.replay()
        return self.__popen(command, None, False)

    def view(self):
        from pty import spawn

        command = self.command.view()
        spawn(command)

    def getpid(self, pid: int) -> int:
        return self.command.getpid(pid)


@dataclass
class Launcher:
    from subprocess import Popen
    from socket import socket

    executor: Executor

    def __init__(self, command: Command):
        self.executor = Executor(command)

    def run(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Popen:
        return self.executor.run(env=env, aslr=aslr, redirect=redirect)

    def debug(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Popen:
        popen = self.executor.debug(env=env, aslr=aslr, redirect=redirect)
        self.executor.view()
        return popen

    def attach(self, pid: int) -> Popen:
        popen = self.executor.attach(pid)
        self.executor.view()
        return popen

    def record(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Popen:
        return self.executor.record(env=env, aslr=aslr, redirect=redirect)

    def replay(self):
        with self.executor.replay() as popen:
            self.executor.view()
            popen.wait()

    def getpid(self, pid: int) -> int:
        return self.executor.getpid(pid)


class Pclose:
    from subprocess import Popen
    TIMEOUT: float = 0.1

    def __init__(self, popen: Popen):
        from subprocess import Popen

        self.__popen: Popen = popen

    def kill(self):
        from contextlib import suppress
        from os import kill
        from signal import SIGINT, SIGTERM, SIGKILL
        from subprocess import TimeoutExpired

        if self.__popen.poll() is None:
            for signal in [SIGINT, SIGTERM, SIGKILL]:
                kill(self.__popen.pid, signal)

                with suppress(TimeoutExpired):
                    self.__popen.wait(self.TIMEOUT)
                    break

    def __enter__(self):
        pass

    def __exit__(self, *_):
        self.kill()


@dataclass
class Container:
    docker: Docker

    def remove(self):
        from subprocess import run, DEVNULL

        command = [self.docker.alias.docker, 'rm', '-f', self.docker.name]
        stdin = stdout = stderr = DEVNULL
        run(command, stdin=stdin, stdout=stdout, stderr=stderr)

    def __enter__(self):
        self.remove()

    def __exit__(self, *_):
        self.remove()


class Hexdump:
    @staticmethod
    def lines(data: bytes) -> Iterator[str]:
        def dumpq(data: bytes) -> str:
            emphasize = [0x0a, 0x55, 0x7f, 0xff]
            text = ''

            for i in range(8):
                if i < len(data):
                    b = data[i]

                    if b in emphasize:
                        text += f'{Color.GREEN}{b:02x}{Color.END}'
                    else:
                        text += f'{b:02x}'
                else:
                    text += '  '

            return text

        def dumps(data: bytes) -> str:
            text = bytearray(data)

            for i, b in enumerate(text):
                if not ord(' ') <= b <= ord('~'):
                    text[i] = ord('.')

            return text.decode()

        offset = 0

        while data:
            fst, snd, data = data[:8], data[8:16], data[16:]
            left, middle, right = dumpq(fst), dumpq(snd), dumps(fst+snd)
            yield f'{Color.CYAN}[{offset:04x}]{Color.END} {left} {middle} {right}'
            offset += 0x10

    @staticmethod
    def border(char: str) -> str:
        return char*57


class Tube(metaclass=ABCMeta):
    @abstractmethod
    def send(self, data: bytes) -> int:
        pass

    @abstractmethod
    def recv(self) -> bytes:
        pass

    @abstractmethod
    def close(self):
        pass


class SocketTube(Tube):
    from socket import socket

    INTERVAL: float = 0.1
    RECVSZ: int = 0x1000

    def __init__(self, sk: socket):
        from socket import socket

        sendsk, recvsk = sk, sk.dup()
        sendsk.settimeout(None)
        recvsk.settimeout(self.INTERVAL)
        self.__sendsk: socket = sendsk
        self.__recvsk: socket = recvsk

    def send(self, data: bytes) -> int:
        return self.__sendsk.send(data)

    def recv(self) -> bytes:
        from contextlib import suppress

        data = b''

        with suppress(TimeoutError):
            data = self.__recvsk.recv(self.RECVSZ)

        return data

    def close(self):
        self.__sendsk.close()
        self.__recvsk.close()


class Logger(Tube):
    SENDHDR: str = '  >'
    RECVHDR: str = '    <'

    def __init__(self, tube: Tube, verbose: int):
        self.__tube: Tube = tube
        self.__verbose: int = verbose

    @staticmethod
    def __hexdump(data: bytes, header: str) -> str:
        text = ''

        for line in Hexdump.lines(data):
            text += f'{header} {line}\n'

        if text:
            border = Hexdump.border('-')
            text = f'{header} {border}\n'+text

        return text

    def __print(self, data: bytes, header: str):
        from os import write
        from sys import stderr

        message = b''

        if self.__verbose == 1:
            message = data

        if self.__verbose >= 2:
            message = self.__hexdump(data, header).encode()

        write(stderr.fileno(), message)

    def send(self, data: bytes) -> int:
        n = self.__tube.send(data)
        data = data[:n]
        self.__print(data, self.SENDHDR)
        return n

    def recv(self) -> bytes:
        data = self.__tube.recv()
        self.__print(data, self.RECVHDR)
        return data

    def close(self):
        self.__tube.close()


class Proxy(Thread):
    def __init__(self, tube: Tube):
        from threading import Event
        from queue import Queue

        super().__init__()
        self.__tube: Tube = tube
        self.__event: Event = Event()
        self.__queue: Queue = Queue()
        self.__buffer: bytes = b''
        self.start()

    def run(self):
        while not self.__event.is_set():
            if data := self.__tube.recv():
                self.__queue.put(data)

    def stop(self):
        self.__event.set()
        self.join()
        self.__tube.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_):
        self.stop()

    def send(self, data: bytes):
        while data:
            n = self.__tube.send(data)
            data = data[n:]

    def sendline(self, data: bytes):
        return self.send(data+b'\n')

    def recv(self, *, size: int = -1, timeout: float = -1) -> bytes:
        from contextlib import suppress
        from queue import Empty

        with suppress(Empty):
            while not self.__queue.empty():
                self.__buffer += self.__queue.get(block=False)

        if timeout and not self.__buffer:
            try:
                timeout_ = timeout if timeout > 0 else None
                self.__buffer = self.__queue.get(block=True, timeout=timeout_)
            except Empty:
                raise TimeoutError('recv timed out')

        size = size if size >= 0 else len(self.__buffer)
        data, self.__buffer = self.__buffer[:size], self.__buffer[size:]
        return data

    def __cancel(self, data: bytes):
        self.__buffer = data+self.__buffer

    def recvcond(self, cond: Callable[[bytes], int], *, timeout: float = -1) -> bytes:
        data = b''

        while True:
            try:
                data += self.recv(timeout=timeout)
            except:
                self.__cancel(data)
                raise

            if (pos := cond(data)) >= 0:
                data, rest = data[:pos], data[pos:]
                self.__cancel(rest)
                return data

            if not timeout:
                self.__cancel(data)
                return b''

    def recvuntil(self, delim: bytes, *, timeout: float = -1) -> bytes:
        def cond(data: bytes) -> int:
            if (pos := data.find(delim)) != -1:
                pos += len(delim)
                return pos
            else:
                return -1

        return self.recvcond(cond, timeout=timeout)

    def recvexact(self, n: int, *, timeout: float = -1) -> bytes:
        def cond(data: bytes) -> int:
            if len(data) >= n:
                return n
            else:
                return -1

        return self.recvcond(cond, timeout=timeout)

    def recvline(self, *, timeout: float = -1) -> bytes:
        return self.recvuntil(b'\n', timeout=timeout)

    def sendafter(self, delim: bytes, data: bytes, *, timeout: float = -1):
        self.recvuntil(delim, timeout=timeout)
        self.send(data)

    def sendlineafter(self, delim: bytes, data: bytes, *, timeout: float = -1):
        self.recvuntil(delim, timeout=timeout)
        self.sendline(data)

    def interactive(self):
        from contextlib import suppress

        with suppress(KeyboardInterrupt):
            while True:
                data = input().encode()
                self.sendline(data)


class Context:
    from contextlib import ExitStack
    from socket import socket

    type Attach = Callable[[], None]
    type Connect = Callable[[], Tube]
    VERBOSE: int = 1

    @classmethod
    def __create(cls, start: Callable[[ExitStack, Launcher, socket | None], int], attachable: bool,
                 command: Command, connect: Connect | None, verbose: int) -> Self:
        from contextlib import ExitStack
        from socket import socketpair

        stack = ExitStack()

        try:
            if docker := command.lookup(Docker):
                container = Container(docker)
                stack.enter_context(container)

            launcher = Launcher(command)

            if connect:
                pid = start(stack, launcher, None)
                tube = connect()

            else:
                sk, redirect = socketpair()

                try:
                    pid = start(stack, launcher, redirect)
                except:
                    sk.close()
                    raise
                finally:
                    redirect.close()

                tube = SocketTube(sk)

            tube = Logger(tube, verbose)
            proxy = Proxy(tube)
            stack.enter_context(proxy)
            pid = launcher.getpid(pid)

            def attach(pid: int = pid):
                popen = launcher.attach(pid)
                stack.enter_context(popen)
                pclose = Pclose(popen)
                stack.enter_context(pclose)

            def dummy(pid: int = pid):
                pass

            attach = attach if attachable else dummy
            return cls(proxy, attach, stack)
        except:
            stack.close()
            raise

    @classmethod
    def run(cls, command: Command, *, env: dict[str, str] = {}, aslr: bool = True,
            connect: Connect | None = None, verbose: int = VERBOSE) -> Self:
        from contextlib import ExitStack
        from socket import socket

        def start(stack: ExitStack, launcher: Launcher, redirect: socket | None) -> int:
            popen = launcher.run(env=env, aslr=aslr, redirect=redirect)
            stack.enter_context(popen)
            pclose = Pclose(popen)
            stack.enter_context(pclose)
            return popen.pid

        return cls.__create(start, True, command, connect, verbose)

    @classmethod
    def debug(cls, command: Command, *, env: dict[str, str] = {}, aslr: bool = True,
              connect: Connect | None = None, verbose: int = VERBOSE) -> Self:
        from contextlib import ExitStack
        from socket import socket

        if command.lookup(RR):
            def start(stack: ExitStack, launcher: Launcher, redirect: socket | None) -> int:
                stack.callback(launcher.replay)
                popen = launcher.record(env=env, aslr=aslr, redirect=redirect)
                stack.enter_context(popen)
                pclose = Pclose(popen)
                stack.enter_context(pclose)
                return popen.pid

        elif command.lookup(Gdb):
            def start(stack: ExitStack, launcher: Launcher, redirect: socket | None) -> int:
                popen = launcher.debug(env=env, aslr=aslr, redirect=redirect)
                stack.enter_context(popen)
                pclose = Pclose(popen)
                stack.enter_context(pclose)
                return popen.pid

        else:
            raise NotImplementedError

        return cls.__create(start, False, command, connect, verbose)

    def __init__(self, proxy: Proxy, attach: Attach, stack: ExitStack):
        from contextlib import ExitStack

        self.proxy: Proxy = proxy
        self.attach: Context.Attach = attach
        self.__stack: ExitStack = stack

    def close(self):
        self.__stack.close()

    def __enter__(self) -> tuple[Proxy, Attach]:
        return (self.proxy, self.attach)

    def __exit__(self, *args):
        self.__stack.__exit__(*args)


class Limit:
    INT8_MIN: int = -(1 << 7)
    INT16_MIN: int = -(1 << 15)
    INT32_MIN: int = -(1 << 31)
    INT64_MIN: int = -(1 << 63)
    UINT8_MAX: int = (1 << 8)-1
    UINT16_MAX: int = (1 << 16)-1
    UINT32_MAX: int = (1 << 32)-1
    UINT64_MAX: int = (1 << 64)-1


type ByteOrder = Literal['little', 'big']


def p8(value: int) -> bytes:
    assert (Limit.INT8_MIN <= value <= Limit.UINT8_MAX)
    return (value & Limit.UINT8_MAX).to_bytes(length=1)


def p16(value: int, *, byteorder: ByteOrder = 'little') -> bytes:
    assert (Limit.INT16_MIN <= value <= Limit.UINT16_MAX)
    return (value & Limit.UINT16_MAX).to_bytes(length=2, byteorder=byteorder)


def p32(value: int, *, byteorder: ByteOrder = 'little') -> bytes:
    assert (Limit.INT32_MIN <= value <= Limit.UINT32_MAX)
    return (value & Limit.UINT32_MAX).to_bytes(length=4, byteorder=byteorder)


def p64(value: int, *, byteorder: ByteOrder = 'little') -> bytes:
    assert (Limit.INT64_MIN <= value <= Limit.UINT64_MAX)
    return (value & Limit.UINT64_MAX).to_bytes(length=8, byteorder=byteorder)


def pf(value: float, *, byteorder: ByteOrder = 'little') -> bytes:
    from struct import pack

    fmt = dict(little='<f', big='>f')
    return pack(fmt[byteorder], value)


def pd(value: float, *, byteorder: ByteOrder = 'little') -> bytes:
    from struct import pack

    fmt = dict(little='<d', big='>d')
    return pack(fmt[byteorder], value)


def u8(data: bytes, *, signed: bool = False) -> int:
    assert (len(data) == 1)
    return int.from_bytes(data, signed=signed)


def u16(data: bytes, *, signed: bool = False, byteorder: ByteOrder = 'little') -> int:
    assert (len(data) == 2)
    return int.from_bytes(data, signed=signed, byteorder=byteorder)


def u32(data: bytes, *, signed: bool = False, byteorder: ByteOrder = 'little') -> int:
    assert (len(data) == 4)
    return int.from_bytes(data, signed=signed, byteorder=byteorder)


def u64(data: bytes, *, signed: bool = False, byteorder: ByteOrder = 'little') -> int:
    assert (len(data) == 8)
    return int.from_bytes(data, signed=signed, byteorder=byteorder)


def uf(data: bytes, *, byteorder: ByteOrder = 'little') -> int:
    from struct import unpack

    assert (len(data) == 4)
    fmt = dict(little='<f', big='>f')
    return unpack(fmt[byteorder], data)[0]


def ud(data: bytes, *, byteorder: ByteOrder = 'little') -> int:
    from struct import unpack

    assert (len(data) == 8)
    fmt = dict(little='<d', big='>d')
    return unpack(fmt[byteorder], data)[0]


def block(size: int, *pair: tuple[int, bytes]) -> bytes:
    dst = bytearray(size)

    for (i, src) in pair:
        assert (0 <= i <= i+len(src) <= size)
        dst[i:i+len(src)] = src

    return bytes(dst)


def rol64(value: int, n: int) -> int:
    assert (Limit.INT64_MIN <= value <= Limit.UINT64_MAX)
    assert (-63 <= n <= 63)
    value &= Limit.UINT64_MAX

    if 0 <= n:
        value = (value << n) | (value >> (64-n))
    else:
        value = (value >> (-n)) | (value << (64+n))

    value &= Limit.UINT64_MAX
    return value


def ror64(value: int, n: int) -> int:
    return rol64(value, -n)


class Iota:
    from itertools import cycle
    from string import digits, ascii_letters

    __seed: Iterator[bytes] = cycle(c.encode() for c in digits+ascii_letters)

    @classmethod
    def get(cls) -> bytes:
        return next(cls.__seed)


def iota() -> bytes:
    return Iota.get()
