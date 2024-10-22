from socket import socket
from ssl import SSLContext


def tcp(host: str, port: int, *, ssl: SSLContext | None = None) -> socket:
    from contextlib import suppress
    from socket import socket
    from time import sleep

    sk = socket()

    try:
        if ssl:
            sk = ssl.wrap_socket(sk)

        while True:
            with suppress(ConnectionRefusedError):
                sk.connect((host, port))
                break

            sleep(0.1)

        return sk
    except:
        sk.close()
        raise


def udp(host: str, port: int) -> socket:
    from socket import socket, AF_INET, SOCK_DGRAM

    sk = socket(AF_INET, SOCK_DGRAM)

    try:
        sk.connect((host, port))
    except:
        sk.close()
        raise

    return sk


def unsafeSSLContext() -> SSLContext:
    from ssl import SSLContext, PROTOCOL_TLS_CLIENT, CERT_NONE

    ssl = SSLContext(PROTOCOL_TLS_CLIENT)
    ssl.check_hostname = False
    ssl.verify_mode = CERT_NONE
    return ssl
