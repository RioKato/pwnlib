from typing import IO, Callable, ClassVar, Iterator, Literal, Self, cast
from abc import ABCMeta, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from threading import Thread
from socket import socket

__all__ = [
    'Alias', 'Target', 'GdbServer', 'RR', 'Qemu', 'Docker', 'Tmux', 'Display',
    'Context', 'Net',
    'p8', 'p16', 'p32', 'p64', 'pf', 'pd',
    'u8', 'u16', 'u32', 'u64', 'uf', 'ud',
    'block',
    'rol64', 'ror64',
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
    def runpid(self, pid: int) -> int:
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

    def runpid(self, pid: int) -> int:
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

    def runpid(self, pid: int) -> int:
        return self.command.runpid(pid)


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
        if aslr:
            raise ValueError

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

    def runpid(self, pid: int) -> int:
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


type Redirect = socket | int | None


@dataclass
class Executor:
    from subprocess import Popen

    command: Command

    def __popen(self, command: list[str], redirect: Redirect, trace: bool) -> Popen:
        from subprocess import Popen

        def pr_set_ptracer_any():
            Glibc.prctl(Glibc.PR_SET_PTRACER, Glibc.PR_SET_PTRACER_ANY, 0, 0, 0)

        stdin = stdout = stderr = cast(IO | int | None, redirect)
        start_new_session = trace
        preexec_fn = pr_set_ptracer_any if trace else None
        return Popen(command, stdin=stdin, stdout=stdout, stderr=stderr,
                     start_new_session=start_new_session, preexec_fn=preexec_fn)

    def run(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: Redirect = None) -> Popen:
        command = self.command.run(env, aslr)
        return self.__popen(command, redirect, True)

    def debug(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: Redirect = None) -> Popen:
        command = self.command.debug(env, aslr)
        return self.__popen(command, redirect, False)

    def attach(self, pid: int) -> Popen:
        from subprocess import DEVNULL

        command = self.command.attach(pid)
        return self.__popen(command, DEVNULL, False)

    def record(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: Redirect = None) -> Popen:
        command = self.command.record(env, aslr)
        return self.__popen(command, redirect, False)

    def replay(self) -> Popen:
        from subprocess import DEVNULL

        command = self.command.replay()
        return self.__popen(command, DEVNULL, False)

    def view(self) -> Popen:
        command = self.command.view()
        return self.__popen(command, None, False)


@dataclass
class Launcher(AbstractContextManager):
    from subprocess import Popen

    executor: Executor
    INTERVAL: ClassVar[float] = 0.1

    class ExitError(Exception):
        pass

    def __init__(self, command: Command):
        from contextlib import ExitStack

        self.executor = Executor(command)
        self.__context: ExitStack = ExitStack()

    def __pclose(self, popen: Popen):
        from contextlib import suppress
        from signal import SIGINT, SIGTERM, SIGKILL
        from subprocess import TimeoutExpired

        def callback():
            if popen.poll() is not None:
                return

            for signal in [SIGINT, SIGTERM, SIGKILL]:
                popen.send_signal(signal)

                with suppress(TimeoutExpired):
                    popen.wait(self.INTERVAL)
                    return

        self.__context.enter_context(popen)
        self.__context.callback(callback)

    def run(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: Redirect = None) -> int:
        popen = self.executor.run(env=env, aslr=aslr, redirect=redirect)
        self.__pclose(popen)
        return self.executor.command.runpid(popen.pid)

    def debug(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: Redirect = None):
        popen = self.executor.debug(env=env, aslr=aslr, redirect=redirect)
        self.__pclose(popen)

    def attach(self, pid: int):
        popen = self.executor.attach(pid)
        self.__pclose(popen)

    def record(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: Redirect = None):
        popen = self.executor.record(env=env, aslr=aslr, redirect=redirect)
        self.__pclose(popen)

    def replay(self):
        popen = self.executor.replay()
        self.__pclose(popen)

    def view(self):
        popen = self.executor.view()
        self.__pclose(popen)

    def kill(self):
        self.__context.pop_all().close()

    def exit(self):
        raise self.ExitError

    def close(self):
        self.__context.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> bool | None:
        if args[0] == self.ExitError:
            self.__context.__exit__(None, None, None)
            return True
        else:
            return self.__context.__exit__(*args)


@dataclass
class Container(AbstractContextManager):
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
        return self.__recvsk.recv(self.RECVSZ)

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


class Proxy(Thread, AbstractContextManager):
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
        from contextlib import suppress

        while not self.__event.is_set():
            with suppress(TimeoutError):
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
                timeout = timeout if timeout > 0 else None  # type:ignore
                self.__buffer = self.__queue.get(block=True, timeout=timeout)
            except Empty as e:
                raise TimeoutError from e

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

        with suppress(KeyboardInterrupt, BrokenPipeError):
            while True:
                data = input().encode()
                self.sendline(data)


class Context:
    type Connect = Callable[[], Tube]
    type Helper = Callable[[], None]

    def __init__(self, command: Command | None, connect: Connect | None, *,
                 env: dict[str, str] = {}, aslr: bool = True, debug: bool = False,
                 verbose: int = 1):
        from contextlib import ExitStack
        from socket import socket, socketpair
        from subprocess import DEVNULL
        from signal import sigwait, SIGINT

        if not command and not connect:
            raise ValueError

        context = ExitStack()

        try:
            def helper():
                pass

            if connect:
                skt, redirect = None, DEVNULL
            else:
                skt, redirect = socketpair()

            if command:
                if docker := command.lookup(Docker):
                    docker = Container(docker)
                    context.enter_context(docker)

                launcher = Launcher(command)
                context.enter_context(launcher)

                try:
                    if debug:
                        if command.lookup(GdbServer) or command.lookup(Qemu):
                            launcher.debug(env=env, aslr=aslr, redirect=redirect)
                            launcher.view()
                        elif command.lookup(RR):
                            launcher.record(env=env, aslr=aslr, redirect=redirect)

                            def helper():
                                launcher.kill()
                                launcher.replay()
                                launcher.view()
                                sigwait([SIGINT])
                                launcher.exit()
                        else:
                            raise NotImplementedError
                    else:
                        pid = launcher.run(env=env, aslr=aslr, redirect=redirect)

                        if command.lookup(GdbServer):
                            def helper():
                                launcher.attach(pid)
                                launcher.view()
                except:
                    if isinstance(skt, socket):
                        skt.close()
                    raise
                finally:
                    if isinstance(redirect, socket):
                        redirect.close()

            if connect:
                tube = connect()
            else:
                assert (skt)
                tube = SocketTube(skt)

            tube = Logger(tube, verbose)
            proxy = Proxy(tube)
            context.enter_context(proxy)
            self.__context: ExitStack = context
            self.proxy: Proxy = proxy
            self.helper: Context.Helper = helper
        except:
            context.close()
            raise

    def close(self):
        self.__context.close()

    def __enter__(self) -> tuple[Proxy, Helper]:
        return (self.proxy, self.helper)

    def __exit__(self, *args) -> bool | None:
        return self.__context.__exit__(*args)


class Net:
    from ssl import SSLContext

    INTERVAL: float = 0.1

    @classmethod
    def tcp(cls, host: str, port: int, *, ssl: SSLContext | None = None) -> Tube:
        from contextlib import suppress
        from socket import socket
        from time import sleep

        sk = socket()

        try:
            if ssl:
                sk = ssl.wrap_socket(sk)

            while True:
                with suppress(ConnectionRefusedError):
                    sk.connect((host, port))
                    break

                sleep(cls.INTERVAL)

            return SocketTube(sk)
        except:
            sk.close()
            raise

    @staticmethod
    def udp(host: str, port: int) -> Tube:
        from socket import socket, AF_INET, SOCK_DGRAM

        sk = socket(AF_INET, SOCK_DGRAM)

        try:
            sk.connect((host, port))
        except:
            sk.close()
            raise

        return SocketTube(sk)

    @staticmethod
    def SSLNoVerify() -> SSLContext:
        from ssl import SSLContext, PROTOCOL_TLS_CLIENT, CERT_NONE

        context = SSLContext(PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = CERT_NONE
        return context


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
