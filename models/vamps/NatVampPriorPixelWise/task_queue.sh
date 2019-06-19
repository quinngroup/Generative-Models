#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python NatVampPrior.py --save ../../weights/exp10/w1 --log obs3/w1; 
python NatVampPriorPixelwise_1.py --save ../../weights/exp10/w2 --log obs3/w2; 
python NatVampPriorPixelwise_2.py --save ../../weights/exp10/w3 --log obs3/w3; 
python NatVampPriorPixelwise_First.py --save ../../weights/exp10/w4 --log obs3/w4; 
python NatVampPriorPixelwise_FirstLast.py --save ../../weights/exp10/w5 --log obs3/w5; 
python NatVampPriorPixelwise_Last.py --save ../../weights/exp10/w6 --log obs3/w6; 