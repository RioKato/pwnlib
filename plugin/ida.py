from dataclasses import dataclass
from pwnlib import Command, Target, Outer, Gdb


@dataclass
class Ida(Outer):
    ida: str

    def __init__(self, command: Command, ida: str = 'ida64'):
        assert (command.lookup(Gdb))
        self.command = command
        self.ida = ida

    def cli(self) -> list[str]:
        target = self.command.lookup(Target)
        assert (target and target.command)
        gdb = self.command.lookup(Gdb)
        assert (gdb)
        host = gdb.host if gdb.host else 'localhost'
        file = gdb.file if gdb.file else target.command[0]
        command = [self.ida, f'-rgdb@{host}:{gdb.port}', file]
        return command
