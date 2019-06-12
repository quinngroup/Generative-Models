#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python VtPVAE.py  --epochs 30 --lsdim 12 --save weights/exp6/w1.h5 --pseudos 25;
python vvtp.py  --epochs 30 --lsdim 12 --save weights/exp6/w2.h5 --pseudos 25;
