#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python vvtp.py  --epochs 20 --lsdim 12 --save weights/exp6/w2.h5 --pseudos 25;
