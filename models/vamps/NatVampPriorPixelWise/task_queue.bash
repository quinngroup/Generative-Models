#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python NatVampPrior --save ../../weights/exp5/w; 
python NatVampPriorPixelwise_1 --save ../../weights/exp10/w1; 
python NatVampPriorPixelwise_2 --save ../../weights/exp10/w2; 
python NatVampPriorPixelwise_First --save ../../weights/exp10/wF; 
python NatVampPriorPixelwise_FirstLast --save ../../weights/exp10/wFL; 
python NatVampPriorPixelwise_Last --save ../../weights/exp10/wL; 