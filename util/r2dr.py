from typing import Protocol


class ELF(Protocol):
    symtab: int
    strtab: int
    rela_plt: int


def r2dr64le(fun: dict[str, int], dummy: int, elf: ELF) -> tuple[dict[str, int], bytes]:
    assert (dummy >= elf.rela_plt and dummy >= elf.symtab and dummy >= elf.strtab)
    from struct import calcsize, pack_into

    relfmt = '<3Q'
    symfmt = '<I2BH2Q'
    relsz = calcsize(relfmt)
    symsz = calcsize(symfmt)

    def packrel(buf: bytearray, offset: int, r_offset: int, symidx: int):
        r_info = (symidx << 32) | 0x7
        return pack_into(relfmt, buf, offset, r_offset, r_info, 0)

    def packsym(buf: bytearray, offset: int, st_name: int):
        return pack_into(symfmt, buf, offset, st_name, 0x12, 0, 0, 0, 0)

    def packstr(buf: bytearray, offset: int, data: str):
        buf[offset:offset + len(data)] = data.encode()

    tail = dummy
    relidx = (tail - elf.rela_plt + relsz - 1) // relsz
    tail = elf.rela_plt + relsz * (relidx + len(fun))
    symidx = (tail - elf.symtab + symsz - 1) // symsz
    tail = elf.symtab + symsz * (symidx + len(fun))
    stridx = tail - elf.strtab
    tail = tail + sum([len(k) + 1 for k in fun.keys()])

    relpos = {}
    buf = bytearray(tail - dummy)
    straddr = elf.strtab + stridx
    for i, (name, r_offset) in enumerate(fun.items()):
        relpos[name] = relidx + i
        packrel(buf, elf.rela_plt + relsz * (relidx + i) - dummy, r_offset, symidx + i)
        packsym(buf, elf.symtab + symsz * (symidx + i) - dummy, straddr - elf.strtab)
        packstr(buf, straddr - dummy, name + '\x00')
        straddr += len(name) + 1
    buf = bytes(buf)

    return (relpos, buf)
