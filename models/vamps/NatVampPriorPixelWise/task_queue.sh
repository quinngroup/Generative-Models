#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python NatVampPrior.py --save ../../weights/exp10/w1 --log obs3/w1_e2 --epochs 80 --lr 1e-4; 
python NatVampPriorPixelwise_1.py --save ../../weights/exp10/w2 --log obs3/w2_e2 --epochs 80 --lr 1e-4; 
python NatVampPriorPixelwise_2.py --save ../../weights/exp10/w3 --log obs3/w3_e2 --epochs 80 --lr 1e-4; 
python NatVampPriorPixelwise_First.py --save ../../weights/exp10/w4 --log obs3/w4_e2 --epochs 80 --lr 1e-4; 
python NatVampPriorPixelwise_FirstLast.py --save ../../weights/exp10/w5 --log obs3/w5_e2 --epochs 80 --lr 1e-4; 
python NatVampPriorPixelwise_Last.py --save ../../weights/exp10/w6 --log obs3/w6_e2 --epochs 80 --lr 1e-4; 