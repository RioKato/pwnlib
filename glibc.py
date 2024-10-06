from pwnlib import rol64, ror64


def protect(ptr: int, pos: int) -> int:
    return ptr ^ (pos >> 12)


def reveal(ptr: int, pos: int) -> int:
    return protect(ptr, pos)


def mangle(ptr: int, guard: int) -> int:
    return rol64(ptr ^ guard, 0x11)


def demangle(ptr: int, guard: int) -> int:
    return ror64(ptr, 0x11) ^ guard
