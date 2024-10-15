from dataclasses import dataclass, field
from pwnlib import Outer


@dataclass
class Docker(Outer):
    name: str
    options: list[str] = field(default_factory=list)
    docker: str = 'docker'
    sh: str = 'sh'
    pidfile: str = '/var/run/run.pid'

    def __exec(self, inner: list[str], *args: str) -> list[str]:
        command = [self.docker, 'exec']
        command += args
        command += self.options
        command += [self.name]
        command += inner
        return command

    def run(self, env: dict[str, str], aslr: bool) -> list[str]:
        from shlex import join

        command = self.command.run(env, aslr)
        command = join(command)
        command = [self.sh, '-c', f'echo $$ > {self.pidfile}; exec {command};']
        return self.__exec(command, '-i')

    def debug(self, env: dict[str, str], aslr: bool) -> list[str]:
        command = self.command.debug(env, aslr)
        return self.__exec(command, '-i')

    def attach(self, pid: int) -> list[str]:
        pid = f'$(cat {self.pidfile})'  # type:ignore
        command = self.command.attach(pid)
        command = ' '.join(command)
        command = [self.sh, '-c', command]
        return self.__exec(command)

    def record(self, env: dict[str, str], aslr: bool) -> list[str]:
        command = self.command.record(env, aslr)
        return self.__exec(command, '-i')

    def replay(self) -> list[str]:
        command = self.command.replay()
        return self.__exec(command)
