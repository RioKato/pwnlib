def r2dr(fun: dict[str, int],
         dummy: int, jmprel: int, symtab: int, strtab: int, *,
         be: bool = False,
         bit32: bool = False) -> tuple[dict[str, int], bytes]:

    assert (dummy >= jmprel and dummy >= symtab and dummy >= strtab)
    from struct import calcsize, pack_into

    if not bit32:
        relfmt = '<3Q' if not be else '>3Q'
        symfmt = '<I2BH2Q' if not be else '>I2BH2Q'

        def packrel(buf: bytearray, offset: int, r_offset: int, symidx: int):
            r_info = (symidx << 32) | 0x7
            return pack_into(relfmt, buf, offset, r_offset, r_info, 0)

        def packsym(buf: bytearray, offset: int, st_name: int):
            return pack_into(symfmt, buf, offset, st_name, 0x12, 0, 0, 0, 0)
    else:
        relfmt = '<2I' if not be else '>2I'
        symfmt = '<3I2BH' if not be else '>3I2BH'

        def packrel(buf: bytearray, offset: int, r_offset: int, symidx: int):
            r_info = (symidx << 8) | 0x7
            return pack_into(relfmt, buf, offset, r_offset, r_info)

        def packsym(buf: bytearray, offset: int, st_name: int):
            return pack_into(symfmt, buf, offset, st_name, 0, 0, 0x12, 0, 0)

    def packstr(buf: bytearray, offset: int, data: str):
        buf[offset:offset+len(data)] = data.encode()

    relsz = calcsize(relfmt)
    symsz = calcsize(symfmt)

    tail = dummy
    relidx = (tail-jmprel+relsz-1)//relsz
    tail = jmprel+relsz*(relidx+len(fun))
    symidx = (tail-symtab+symsz-1)//symsz
    tail = symtab+symsz*(symidx+len(fun))
    stridx = tail-strtab
    tail = tail+sum([len(i)+1 for i in fun.keys()])

    relpos = {}
    buf = bytearray(tail-dummy)
    straddr = strtab+stridx
    for (i, (name, r_offset)) in enumerate(fun.items()):
        relpos[name] = relidx+i
        packrel(buf, jmprel+relsz*(relidx+i)-dummy, r_offset, symidx+i)
        packsym(buf, symtab+symsz*(symidx+i)-dummy, straddr-strtab)
        packstr(buf, straddr-dummy, name+'\x00')
        straddr += len(name)+1
    buf = bytes(buf)

    return (relpos, buf)
