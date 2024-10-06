#!/usr/bin/env python3


def gencode(name: str, assign: dict[str, int]) -> str:
    from ast import AnnAssign, ClassDef, Constant, Load, Module, Name, Store, unparse

    def annassign(name: str, value: int) -> AnnAssign:
        return AnnAssign(target=Name(id=name, ctx=Store()),
                         annotation=Name(id='int', ctx=Load()),
                         value=Constant(value=value),
                         simple=1)

    def classdef(name: str, assign: dict[str, int]) -> ClassDef:
        return ClassDef(name=name,
                        bases=[],
                        keywords=[],
                        body=[annassign(k, v) for k, v in assign.items()],
                        decorator_list=[])

    def module(name: str, assign: dict[str, int]) -> Module:
        return Module(
            body=[classdef(name, assign)],
            type_ignores=[])

    return unparse(module(name, assign))


def printdupkey(k: str):
    from sys import stderr

    RED = '\033[31m'
    END = '\033[0m'
    print(f'{RED}[Duplicated Key]{END} {k}', file=stderr)


def sanitize(name: str) -> str:
    from re import sub
    from keyword import kwlist

    name = sub(r'@.*', '', name)
    name = sub(r'[^0-9a-zA-Z_]', '_', name)

    while '__' in name:
        name = name.replace('__', '_')

    name = name.rstrip('_')

    if name in kwlist:
        name = f'_{name}'

    if name and '0' <= name[0] <= '9':
        name = f'_{name}'

    return name


def debuginfod_find(path: str) -> str:
    from subprocess import run

    command = ['debuginfod-find', 'debuginfo', path]
    return run(command, capture_output=True, text=True).stdout.strip()


def nm(path: str, *options: str) -> dict[str, int]:
    from subprocess import run
    from re import findall, MULTILINE

    command = ['nm', '--defined']
    command += options
    command += [path]
    result = run(command, capture_output=True, text=True, check=True).stdout
    pattern = r'^(\S+)\s+\S+\s+(.+)$'
    syms = {}

    for v, k in findall(pattern, result, MULTILINE):
        k = sanitize(k)
        v = int(v, 16)

        if k in syms and v != syms[k]:
            printdupkey(k)

        syms[k] = v

    return syms


def objdump(path: str) -> dict[str, int]:
    from subprocess import run
    from re import findall, MULTILINE

    command = ['objdump', '-h', path]
    result = run(command, capture_output=True, text=True, check=True).stdout
    pattern = r'^\s*\d+\s+\.(\S+)\s+\S+\s+(\S+)\s+\S+\s+\S+\s+\S+$'
    secs = {}

    for k, v in findall(pattern, result, MULTILINE):
        k = sanitize(k)
        v = int(v, 16)

        if k in secs and v != secs[k]:
            printdupkey(k)

        secs[k] = v

    return secs


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-d', '--demangle', action='store_true')
    parser.add_argument('--disable-debuginfod', action='store_true')
    parser.add_argument('path')
    parser.add_argument('name')
    args = parser.parse_args()

    demangle = ['-C'] if args.demangle else []
    defs = []
    defs.append(nm(args.path, *demangle))
    defs.append(nm(args.path, '-D', *demangle))

    if not args.disable_debuginfod:
        if debuginfo := debuginfod_find(args.path):
            defs.append(nm(debuginfo, *demangle))
            defs.append(nm(debuginfo, '-D', *demangle))

    defs.append(objdump(args.path))
    assign = {}

    for d in defs:
        for k in assign.keys() & d.keys():
            if assign[k] != d[k]:
                printdupkey(k)

        assign |= d

    code = gencode(args.name, assign)
    print(code)


if __name__ == '__main__':
    main()
