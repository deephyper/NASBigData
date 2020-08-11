import os
import pathlib
import argparse
import json
import time
import socket


def hostnames_to_ips(hostnames: str) -> list:
    """3826-3827,3830,3832-3833,3836,3838-3839

    Args:
        hostnames (str): "3826-3827,3830,3832-3833,3836,3838-3839"

    Returns:
        list: generator of ip addresses.
    """
    hostnames = hostnames.split(",")

    def to_nid(hn):
        hn = int(hn) if type(hn) is str else hn
        return f"nid{hn:05d}"

    def addresses_generator(hostnames: list) -> str:
        for hn in hostnames:
            if "-" in hn:
                start, end = hn.split("-")
                for hn_ in range(int(start), int(end) + 1):
                    yield socket.gethostbyname(to_nid(hn_))
            else:
                yield socket.gethostbyname(to_nid(hn))

    return addresses_generator(hostnames)


# Retriving COBALT infos
hostnames = os.environ.get("COBALT_PARTNAME", "")
jobsize = os.environ.get("COBALT_JOBSIZE", 0)

# Building list of ip addresses
ips = list(hostnames_to_ips(hostnames))

print(ips)

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)
print("My IP is: ", IPAddr)
