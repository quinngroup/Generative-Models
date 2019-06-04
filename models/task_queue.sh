#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python VtPVAE.py --epochs 40 --lsdim 12 --batch-size 1000 --save weights/exp1/bs1000.h5; python VtPVAE.py --epochs 40 --lsdim 12 --save weights/exp1/bs128.h5

