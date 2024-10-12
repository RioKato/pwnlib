#!/usr/bin/env python3


def gencode(name: str, assign: dict[str, list[int]]) -> str:
    from ast import AnnAssign, ClassDef, Constant, List, Load, Module, Name, Store, Subscript, unparse

    def annassign(name: str, values: list[int]) -> AnnAssign:
        if len(values) == 1:
            return AnnAssign(target=Name(id=name, ctx=Store()),
                             annotation=Name(id='int', ctx=Load()),
                             value=Constant(value=values[0]),
                             simple=1)
        else:
            return AnnAssign(target=Name(id=name, ctx=Store()),
                             annotation=Subscript(value=Name(id='list', ctx=Load()), slice=Name(id='int', ctx=Load()), ctx=Load()),
                             value=List(elts=[Constant(value=v) for v in values], ctx=Load()),
                             simple=1)

    def classdef(name: str, assign: dict[str, list[int]]) -> ClassDef:
        return ClassDef(name=name,
                        bases=[],
                        keywords=[],
                        body=[annassign(k, v) for k, v in assign.items()],
                        decorator_list=[],
                        type_params=[])

    def module(name: str, assign: dict[str, list[int]]) -> Module:
        return Module(
            body=[classdef(name, assign)],
            type_ignores=[])

    return unparse(module(name, assign))


def debuginfod_find(path: str) -> str:
    from subprocess import run

    command = ['debuginfod-find', 'debuginfo', path]
    return run(command, capture_output=True, text=True).stdout.strip()


def nm(path: str, *options: str) -> list[tuple[str, int]]:
    from subprocess import run
    from re import findall, MULTILINE

    command = ['nm', '--defined']
    command += options
    command += [path]
    result = run(command, capture_output=True, text=True, check=True).stdout
    pattern = r'^(\S+)\s+\S+\s+(.+)$'
    return [(k, int(v, 16)) for v, k in findall(pattern, result, MULTILINE)]


def readelf(path: str) -> list[tuple[str, int]]:
    from subprocess import run
    from re import findall, MULTILINE

    command = ['readelf', '-S', '-W', path]
    result = run(command, capture_output=True, text=True, check=True).stdout
    pattern = r'^\s*\[\s*\d+\]\s+\.(\S+)\s+\S+\s+(\S+).*$'
    return [(k, int(v, 16)) for k, v in findall(pattern, result, MULTILINE)]


def sanitize(name: str) -> str:
    from re import sub
    from keyword import kwlist

    block = list(kwlist)
    block += ['list', 'int']
    name = sub(r'@.*', '', name)
    name = sub(r'[^0-9a-zA-Z_]', '_', name)
    name = sub(r'_+', '_', name)
    name = name.rstrip('_')
    name = name if name not in block else f'_{name}'
    name = name if name else '_'
    name = name if not '0' <= name[0] <= '9' else f'_{name}'
    return name


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-d', '--demangle', action='store_true')
    parser.add_argument('--disable-debuginfod', action='store_true')
    parser.add_argument('path')
    parser.add_argument('name')
    args = parser.parse_args()

    demangle = ['-C'] if args.demangle else []
    syms = []
    syms += nm(args.path, *demangle)
    syms += nm(args.path, '-D', *demangle)

    if not args.disable_debuginfod:
        if debuginfo := debuginfod_find(args.path):
            syms += nm(debuginfo, *demangle)
            syms += nm(debuginfo, '-D', *demangle)

    syms += readelf(args.path)
    pysyms = {}

    for k, v in syms:
        k = sanitize(k)

        if k not in pysyms:
            pysyms[k] = set()

        pysyms[k].add(v)

    pysyms = dict((k, sorted(list(v))) for k, v in pysyms.items())
    code = gencode(args.name, pysyms)
    print(code)


if __name__ == '__main__':
    main()
