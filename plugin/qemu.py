from dataclasses import dataclass
from pwnlib import Target, Gdb


@dataclass
class Qemu(Gdb):
    def __post_init__(self):
        if target := self.command.lookup(Target):
            if target.command:
                self.file = self.file if self.file else target.command[0]

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
