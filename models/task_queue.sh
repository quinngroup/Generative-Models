#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python VtPVAE.py --epochs 20 --lsdim 12 --batch-size 1000 --gamma .05 --save weights/exp2/g05.h5; python VtPVAE.py --epochs 20 --lsdim 12 --gamma .1 --save weights/exp2/g10.h5; python VtPVAE.py --epochs 20 --lsdim 12 --gamma .15 --save weights/exp2/g15.h5

