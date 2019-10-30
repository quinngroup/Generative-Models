#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)


python 3DConv.py --epochs 30 --batch-size 50 --log obs7/r0 --save weights/exp14/r0.h5;
