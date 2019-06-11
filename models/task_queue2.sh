#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python 3DConv.py  --epochs 30 --lsdim 16 --save weights/3dconv/w.h5 --pseudos 25;
