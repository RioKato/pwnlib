from typing import IO, Callable, Iterator, Literal, Protocol, Self, cast
from abc import ABCMeta, abstractmethod
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from subprocess import Popen
from multiprocessing import Process

__all__ = [
    'Alias', 'Target', 'GdbServer', 'RR', 'Tmux', 'Display',
    'Setup', 'Net',
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
    env: str = 'env'
    setarch: str = 'setarch'
    gdb: str = 'gdb'
    gdbserver: str = 'gdbserver'
    rr: str = 'rr'
    tmux: str = 'tmux'


class Command(metaclass=ABCMeta):
    @property
    @abstractmethod
    def alias(self) -> Alias:
        pass

    @abstractmethod
    def lookup[T: Command](self, cls: type[T]) -> T | None:
        pass

    def run(self, env: dict[str, str], aslr: bool) -> list[str]:
        raise NotImplementedError

    def debug(self, env: dict[str, str], aslr: bool) -> list[str]:
        raise NotImplementedError

    def attach(self, pid: int) -> list[str]:
        raise NotImplementedError

    def record(self, env: dict[str, str], aslr: bool) -> list[str]:
        raise NotImplementedError

    def replay(self) -> list[str]:
        raise NotImplementedError

    def cli(self) -> list[str]:
        raise NotImplementedError


@dataclass
class Target(Command):
    command: list[str]

    def __init__(self, command: list[str], alias: Alias | None = None):
        self.command = command
        self.__alias: Alias = alias if alias else Alias()

    @property
    def alias(self) -> Alias:
        return self.__alias

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


@dataclass
class Outer(Command):
    command: Command

    @property
    def alias(self) -> Alias:
        return self.command.alias

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

    def cli(self) -> list[str]:
        return self.command.cli()


@dataclass
class Gdb(Outer):
    host: str = ''
    port: int = 1234
    file: str = ''
    sysroot: str = ''
    startup: str = 'target remote {host}:{port}'
    script: str = ''

    def cli(self) -> list[str]:
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
class Tmux(Outer):
    options: list[str] = field(default_factory=list)

    def cli(self) -> list[str]:
        command = [self.alias.tmux, 'split']
        command += self.options
        command += self.command.cli()
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

    def cli(self) -> list[str]:
        command = self.command.cli()
        self.__display(command)
        return command


class Glibc:
    from ctypes import CDLL, c_int, c_ulong

    PR_SET_PTRACER = 0x59616d61
    PR_SET_PTRACER_ANY = -1

    prctl = CDLL(None).prctl
    prctl.restype = c_int
    prctl.argtypes = [c_int, c_ulong, c_ulong, c_ulong, c_ulong]


def popen(command: list[str], trace: bool,
          stdin: IO | int | None, stdout: IO | int | None, stderr: IO | int | None) -> Popen:
    from subprocess import Popen

    def pr_set_ptracer_any():
        Glibc.prctl(Glibc.PR_SET_PTRACER, Glibc.PR_SET_PTRACER_ANY, 0, 0, 0)

    preexec_fn = pr_set_ptracer_any if trace else None
    return Popen(command, stdin=stdin, stdout=stdout, stderr=stderr,
                 process_group=0, preexec_fn=preexec_fn)


@dataclass
class Executor:
    from socket import socket
    from subprocess import Popen

    command: Command

    def run(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Popen:
        from subprocess import DEVNULL

        command = self.command.run(env, aslr)
        stdin = cast(IO, redirect) if redirect else DEVNULL
        stdout = stderr = cast(IO, redirect) if redirect else None
        return popen(command, True, stdin, stdout, stderr)

    def debug(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Popen:
        from subprocess import DEVNULL

        command = self.command.debug(env, aslr)
        stdin = cast(IO, redirect) if redirect else DEVNULL
        stdout = stderr = cast(IO, redirect) if redirect else None
        return popen(command, False, stdin, stdout, stderr)

    def attach(self, pid: int) -> Popen:
        from subprocess import DEVNULL

        command = self.command.attach(pid)
        return popen(command, False, DEVNULL, None, None)

    def record(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Popen:
        from subprocess import DEVNULL

        command = self.command.record(env, aslr)
        stdin = cast(IO, redirect) if redirect else DEVNULL
        stdout = stderr = cast(IO, redirect) if redirect else None
        return popen(command, False, stdin, stdout, stderr)

    def replay(self) -> Popen:
        from subprocess import DEVNULL

        command = self.command.replay()
        return popen(command, False, DEVNULL, None, None)

    def cli(self) -> Popen:
        command = self.command.cli()
        return popen(command, False, None, None, None)


@contextmanager
def pclose(popen: Popen) -> Iterator[Popen]:
    from subprocess import TimeoutExpired
    from signal import SIGTERM, SIGKILL
    from time import sleep

    with popen:
        try:
            yield popen
        finally:
            if popen.poll() is None:
                popen.terminate()

                try:
                    popen.wait(1)
                except TimeoutExpired:
                    popen.kill()

    if popen.returncode not in [0, -SIGTERM, -SIGKILL]:
        sleep(0.5)

    if True:
        from sys import stderr
        from signal import Signals

        code = popen.returncode

        if code >= 0:
            reason = f'pid={popen.pid}, code={code}'
        else:
            signal = Signals(-code).name
            reason = f'pid={popen.pid}, signal={signal}'

        message = f'{Color.RED}[DEBUG]{Color.END} process has terminated({reason})'
        print(message, file=stderr)


class StopRecording(Exception):
    pass


@dataclass
class Context:
    from contextlib import contextmanager
    from socket import socket

    executor: Executor

    @contextmanager
    def run(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[int]:
        with pclose(self.executor.run(env=env, aslr=aslr, redirect=redirect)) as popen:
            yield popen.pid

    @contextmanager
    def debug(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[None]:
        with pclose(self.executor.debug(env=env, aslr=aslr, redirect=redirect)):
            with pclose(self.executor.cli()):
                yield

    @contextmanager
    def attach(self, pid: int) -> Iterator[None]:
        with pclose(self.executor.attach(pid)):
            with pclose(self.executor.cli()):
                yield

    @contextmanager
    def record(self, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[None]:
        from contextlib import suppress
        from signal import sigwait, SIGINT

        try:
            with pclose(self.executor.record(env=env, aslr=aslr, redirect=redirect)):
                with suppress(StopRecording):
                    yield
        finally:
            with pclose(self.executor.replay()):
                with pclose(self.executor.cli()):
                    sigwait([SIGINT])

    def replay(self):
        raise StopRecording


class Launcher:
    from contextlib import contextmanager
    from socket import socket

    @staticmethod
    @contextmanager
    def run(command: Command, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[Callable[[], None]]:
        from contextlib import ExitStack

        context = Context(Executor(command))

        with context.run(env=env, aslr=aslr, redirect=redirect) as pid:
            if context.executor.command.lookup(GdbServer):
                with ExitStack() as estack:
                    yield lambda: estack.enter_context(context.attach(pid))
            else:
                yield lambda: None

    @staticmethod
    @contextmanager
    def debug(command: Command, *, env: dict[str, str] = {}, aslr: bool = True, redirect: socket | None = None) -> Iterator[Callable[[], None]]:
        context = Context(Executor(command))

        if context.executor.command.lookup(RR):
            with context.record(env=env, aslr=aslr, redirect=redirect):
                yield context.replay
        else:
            with context.debug(env=env, aslr=aslr, redirect=redirect):
                yield lambda: None


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
            left, middle, right = dumpq(fst), dumpq(snd), dumps(fst + snd)
            yield f'{Color.CYAN}[{offset:04x}]{Color.END} {left} {middle} {right}'
            offset += 0x10

    @staticmethod
    def border(char: str) -> str:
        return char * 57


class Socket(Protocol):
    def send(self, data: bytes, /) -> int:
        ...

    def recv(self, size: int, /) -> bytes:
        ...

    def close(self):
        ...


class Logger:
    SENDHDR: str = '  >'
    RECVHDR: str = '    <'

    def __init__(self, socket: Socket, verbose: int):
        self.__socket: Socket = socket
        self.__verbose: int = verbose

    @staticmethod
    def __hexdump(data: bytes, header: str) -> str:
        text = ''

        for line in Hexdump.lines(data):
            text += f'{header} {line}\n'

        if text:
            border = Hexdump.border('-')
            text = f'{header} {border}\n' + text

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

    def send(self, data: bytes, /) -> int:
        n = self.__socket.send(data)
        data = data[:n]
        self.__print(data, self.SENDHDR)
        return n

    def recv(self, size: int, /) -> bytes:
        data = self.__socket.recv(size)
        self.__print(data, self.RECVHDR)
        return data

    def close(self):
        self.__socket.close()


class Proxy(Process, AbstractContextManager):
    def __init__(self, socket: Socket, *, verbose: int = 1):
        from multiprocessing import Queue

        socket = Logger(socket, verbose)
        super().__init__()
        self.__socket: Socket = socket
        self.__queue: Queue = Queue()
        self.__eof: bool = False
        self.__buffer: bytes = b''

    def run(self):
        from contextlib import suppress, closing
        from os import setpgid

        setpgid(0, 0)

        with closing(self.__socket):
            with suppress(Exception):
                while data := self.__socket.recv(0x1000):
                    self.__queue.put(data)

            self.__queue.put(b'')

    def stop(self):
        if self.is_alive():
            self.terminate()

        self.join()
        self.close()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_) -> bool | None:
        self.stop()

    def send(self, data: bytes):
        while data:
            n = self.__socket.send(data)
            data = data[n:]

    def sendline(self, data: bytes):
        return self.send(data + b'\n')

    def __recv(self, timeout: float) -> bytes:
        from queue import Empty

        if self.__eof:
            raise EOFError

        block = bool(timeout)
        timeout = timeout if timeout > 0 else None  # type:ignore

        try:
            if data := self.__queue.get(block=block, timeout=timeout):
                return data
            else:
                self.__eof = True
                raise EOFError
        except Empty as e:
            if block:
                raise TimeoutError from e
            else:
                return b''

    def recv(self, *, size: int = -1, timeout: float = -1) -> bytes:
        try:
            while data := self.__recv(0):
                self.__buffer += data

            if timeout and not self.__buffer:
                self.__buffer = self.__recv(timeout)
        except EOFError:
            if not self.__buffer:
                raise

        size = size if size >= 0 else len(self.__buffer)
        data, self.__buffer = self.__buffer[:size], self.__buffer[size:]
        return data

    def __cancel(self, data: bytes):
        self.__buffer = data + self.__buffer

    def recvcond(self, cond: Callable[[bytes], int], *, timeout: float = -1) -> bytes:
        data = b''

        while True:
            try:
                data += self.recv(timeout=timeout)
            except:
                self.__cancel(data)
                raise

            pos = cond(data)

            if pos >= 0:
                data, rest = data[:pos], data[pos:]
                self.__cancel(rest)
                return data

            if not timeout:
                self.__cancel(data)
                return b''

    def recvuntil(self, delim: bytes, *, timeout: float = -1) -> bytes:
        def cond(data: bytes) -> int:
            pos = data.find(delim)

            if pos != -1:
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


class Setup(AbstractContextManager):
    from socket import socket

    def __init__(self, command: Command | None, connect: Callable[[], Socket] | None, debug: bool, *,
                 env: dict[str, str] = {}, aslr: bool = True, verbose: int = 1):
        from contextlib import ExitStack, closing
        from socket import socketpair

        assert (command or connect)
        estack = ExitStack()

        try:
            helper = lambda: None

            if connect:
                socket, redirect = None, None
            else:
                socket, redirect = socketpair()
                estack.enter_context(socket)

            if command:
                try:
                    if debug:
                        context = Launcher.debug(command, env=env, aslr=aslr, redirect=redirect)
                        helper = estack.enter_context(context)
                    else:
                        context = Launcher.run(command, env=env, aslr=aslr, redirect=redirect)
                        helper = estack.enter_context(context)
                finally:
                    if redirect:
                        redirect.close()

            if connect:
                socket = connect()
                estack.enter_context(closing(socket))
            else:
                assert (socket)

            proxy = Proxy(socket, verbose=verbose)
            estack.enter_context(proxy)
            self.__estack: ExitStack = estack
            self.proxy: Proxy = proxy
            self.helper: Callable[[], None] = helper
        except:
            estack.close()
            raise

    def __enter__(self) -> tuple[Proxy, Callable[[], None]]:
        return (self.proxy, self.helper)

    def __exit__(self, *args) -> bool | None:
        return self.__estack.__exit__(*args)


class Net:
    from socket import socket
    from ssl import SSLContext

    @staticmethod
    def tcp(host: str, port: int, *, ssl: SSLContext | None = None) -> socket:
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

                sleep(0.1)

            return sk
        except:
            sk.close()
            raise

    @staticmethod
    def udp(host: str, port: int) -> socket:
        from socket import socket, AF_INET, SOCK_DGRAM

        sk = socket(AF_INET, SOCK_DGRAM)

        try:
            sk.connect((host, port))
        except:
            sk.close()
            raise

        return sk

    @staticmethod
    def SSLNoVerify() -> SSLContext:
        from ssl import SSLContext, PROTOCOL_TLS_CLIENT, CERT_NONE

        ssl = SSLContext(PROTOCOL_TLS_CLIENT)
        ssl.check_hostname = False
        ssl.verify_mode = CERT_NONE
        return ssl


class Limit:
    INT8_MIN: int = -(1 << 7)
    INT16_MIN: int = -(1 << 15)
    INT32_MIN: int = -(1 << 31)
    INT64_MIN: int = -(1 << 63)
    UINT8_MAX: int = (1 << 8) - 1
    UINT16_MAX: int = (1 << 16) - 1
    UINT32_MAX: int = (1 << 32) - 1
    UINT64_MAX: int = (1 << 64) - 1


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


def block(size: int, filler: bytes, *pair: tuple[int, bytes]) -> bytes:
    assert (len(filler) == 1)
    dst = bytearray(filler * size)

    for (i, src) in pair:
        assert (0 <= i <= i + len(src) <= size)
        dst[i:i + len(src)] = src

    return bytes(dst)


def rol64(value: int, n: int) -> int:
    assert (Limit.INT64_MIN <= value <= Limit.UINT64_MAX)
    assert (-63 <= n <= 63)
    value &= Limit.UINT64_MAX

    if 0 <= n:
        value = (value << n) | (value >> (64 - n))
    else:
        value = (value >> (-n)) | (value << (64 + n))

    value &= Limit.UINT64_MAX
    return value


def ror64(value: int, n: int) -> int:
    return rol64(value, -n)


class Iota(bytearray):
    def __init__(self):
        from itertools import cycle
        from string import digits, ascii_letters

        super().__init__()
        self.__seed: Iterator[bytes] = cycle(c.encode() for c in digits + ascii_letters)
        self()

    def __call__(self) -> Self:
        self[:] = next(self.__seed)
        return self


iota: Iota = Iota()
