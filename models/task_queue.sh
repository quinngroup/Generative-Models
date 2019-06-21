#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python 3DConv.py  --epochs 80 --save weights/exp13/p0.h5 --log obs6/p0; 
python 3DConv.py  --epochs 80 --save weights/exp13/p5.h5 --log obs6/p5 --schedule 5; 
python 3DConv.py  --epochs 80 --save weights/exp13/p10.h5 --log obs6/p10 --schedule 10; 
