#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python VtPVAE.py --epochs 20 --lsdim 12 --gamma .05 --save weights/exp2/g05b128.h5

