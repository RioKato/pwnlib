from signal import SIGCONT, Signals


def rr_attach(ppid: int, *, signal: Signals = SIGCONT):
    from re import search
    from os import kill

    while True:
        with open(f'/proc/{ppid}/task/{ppid}/children') as fd:
            text = fd.read()

            if pid := search(r'(\d+)', text):
                pid = pid.group()
                pid = int(pid)
                kill(pid, signal)
                return


def gdb_sigint():
    from subprocess import run
    from signal import SIGINT
    from os import kill
    from re import search, MULTILINE

    command = ['ps', '-o', 'pid=', '--sort=-start_time', '-C', 'gdb']
    pids = run(command, check=True, capture_output=True, text=True).stdout
    found = search(r'(\d+)', pids, MULTILINE)
    assert (found)
    pid = found.group()
    pid = int(pid)
    kill(pid, SIGINT)
