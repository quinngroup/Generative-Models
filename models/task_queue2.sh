#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)


python 3DConv.py --epochs 30 --batch-size 50 --log obs7/r0 --save weights/exp14/r0.h5;
python 3DConv.py --epochs 30 --batch-size 50 --log obs7/r1 --save weights/exp14/r1.h5 --reg2 .01;
python 3DConv.py --epochs 30 --batch-size 50 --log obs7/r2 --save weights/exp14/r2.h5 --reg2 .05;
python 3DConv.py --epochs 30 --batch-size 50 --log obs7/r3 --save weights/exp14/r3.h5 --reg2 .1;
python 3DConv.py --epochs 30 --batch-size 50 --log obs7/r4 --save weights/exp14/r4.h5 --reg2 .2;
python 3DConv.py --epochs 30 --batch-size 50 --log obs7/r5 --save weights/exp14/r5.h5 --reg2 .4;
python 3DConv.py --epochs 30 --batch-size 50 --log obs7/r6 --save weights/exp14/r6.h5 --reg2 .6;
python 3DConv.py --epochs 30 --batch-size 50 --log obs7/r7 --save weights/exp14/r7.h5 --reg2 .8;
