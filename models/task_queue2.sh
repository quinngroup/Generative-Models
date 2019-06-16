#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)


python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 200 --log obs1/b2_l1 --save weights/exp8/b2_l1.h5;

python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 200 --lr 1e-4 --log obs1/b2_l2 --save weights/exp8/b2_l2.h5;

python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 200 --lr 1e-5 --log obs1/b2_l3 --save weights/exp8/b2_l3.h5;

python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 200 --lr 1e-6 --log obs1/b2_l4 --save weights/exp8/b2_l4.h5;

python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 100 --log obs1/b1_l1 --save weights/exp8/b1_l1.h5;

python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 100 --lr 1e-4 --log obs1/b1_l2 --save weights/exp8/b1_l2.h5;

python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 100 --lr 1e-5 --log obs1/b1_l3 --save weights/exp8/b1_l3.h5;

python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 100 --lr 1e-6 --log obs1/b1_l4 --save weights/exp8/b1_l4.h5;



