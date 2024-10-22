from dataclasses import dataclass
from pwnlib import Command, Target, Outer, Gdb


@dataclass
class Ida(Outer):
    ida: str

    def __init__(self, command: Command, ida: str = 'ida64'):
        self.command = command
        self.ida = ida

    def cli(self) -> list[str]:
        gdb = self.command.lookup(Gdb)
        assert (gdb)
        host = gdb.host if gdb.host else 'localhost'
        port = gdb.port
        file = gdb.file

        if not file:
            if target := self.command.lookup(Target):
                if target.command:
                    file = target.command[0]

        command = [self.ida, f'-rgdb@{host}:{port}', file]
        return command
