#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)


python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 7e-5 --log obs5/pl7e5 --save weights/exp12/pl7e5.h5;


