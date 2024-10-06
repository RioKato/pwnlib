#!/usr/bin/env python3

from pathlib import Path


def printsymlink(item: Path):
    from sys import stderr

    RED = '\033[31m'
    END = '\033[0m'
    symlink = item.readlink()

    if item.exists():
        message = f'{RED}[Replaced SymLink]{END} {item}: {symlink}'
    else:
        message = f'{RED}[Replaced SymLink]{END} {item}: Not Found({symlink})'

    print(message, file=stderr)


def rootlink(sysroot: str):
    from contextlib import suppress
    from pathlib import Path
    from os import walk
    from os.path import relpath

    sysroot_ = Path(sysroot)

    for path, dirs, files in walk(sysroot_):
        for item in dirs+files:
            item = Path(path, item)

            if not item.is_symlink():
                continue

            symlink = item.readlink()

            if not symlink.is_absolute():
                continue

            symlink = symlink.relative_to(symlink.root)
            symlink = sysroot_.joinpath(symlink)
            symlink = relpath(symlink, item.parent)

            with suppress(FileNotFoundError):
                item.unlink()

            item.symlink_to(symlink)
            printsymlink(item)


def docker_cpall(image: str) -> str:
    from subprocess import run

    command = ['docker', 'create', image]
    cid = run(command, check=True, capture_output=True, text=True).stdout.strip()

    try:
        out = image.replace(':', '_')
        command = ['docker', 'cp', f'{cid}:/', out]
        run(command, check=True)

    finally:
        command = ['docker', 'rm', '-v', cid]
        run(command, check=True)

    return out


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-d', '--docker', action='store_true')
    parser.add_argument('image')
    args = parser.parse_args()
    out = args.image

    if args.docker:
        out = docker_cpall(out)

    rootlink(out)


if __name__ == '__main__':
    main()
