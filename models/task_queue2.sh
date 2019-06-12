#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python 3DConv.py  --epochs 30 --lsdim 16 --save weights/3dconv/b1.h5 --pseudos 25 --batch-size 64;
python 3DConv.py  --epochs 30 --lsdim 16 --save weights/3dconv/b2.h5 --pseudos 25 --batch-size 128;
python 3DConv.py  --epochs 30 --lsdim 16 --save weights/3dconv/b3.h5 --pseudos 25 --batch-size 256;
