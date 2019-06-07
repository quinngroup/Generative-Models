#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python VtPVAE.py  --epochs 30 --lsdim 16 --gamma .05 --save weights/exp3/b0.h5 --pseudos 25; 

python VtPVAE.py  --epochs 30 --lsdim 16 --gamma .15 --save weights/exp3/b1.h5 --pseudos 25 --batch-size 1000; 

python VtPVAE.py  --epochs 10 --lsdim 16 --gamma .05 --save weights/exp3/_b2.h5 --pseudos 25; 
python VtPVAE.py  --epochs 20 --lsdim 16 --gamma .15 --save weights/exp3/b2.h5 --pseudos 25 --load weights/exp3/_b2.h5 --batch-size 1000; 

python VtPVAE.py  --epochs 20 --lsdim 16 --gamma .05 --save weights/exp3/_b3.h5 --pseudos 25;
python VtPVAE.py  --epochs 10 --lsdim 16 --gamma .15 --save weights/exp3/b3.h5 --pseudos 25 --load weights/exp3/_b3.h5 --batch-size 1000; 