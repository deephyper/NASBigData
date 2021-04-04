#!/bin/bash

git clone --depth=1 --branch=master https://github.com/deephyper/Benchmarks.git
rm -rf Benchmarks/.git/
python Benchmarks/Pilot1/Attn/attn_loader.py