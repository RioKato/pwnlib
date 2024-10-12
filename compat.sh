#!/bin/sh

cd $(dirname $0)

sed -i -e 's/from typing/from typing_extensions/' pwnlib.py
sed -i -e 's/def lookup\[T: Command\](self, cls\: type\[T\]) -> T | None/def lookup(cself, cls)/' pwnlib.py
sed -i -e 's/to_bytes(length=1)/to_bytes(length=1, byteorder="little")/' pwnlib.py
