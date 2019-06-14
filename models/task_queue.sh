#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python VtPVAE.py  --epochs 30 --pseudos 25 --lsdim 2 --save weights/exp5/w1.h5; 
python VtPVAE.py  --epochs 30 --pseudos 25 --lsdim 4 --save weights/exp5/w2.h5; 
python VtPVAE.py  --epochs 30 --pseudos 25 --lsdim 8 --save weights/exp5/w3.h5; 
python VtPVAE.py  --epochs 30 --pseudos 25 --lsdim 16 --save weights/exp5/w4.h5; 
python VtPVAE.py  --epochs 30 --pseudos 25 --lsdim 32 --save weights/exp5/w5.h5; 
python VtPVAE.py  --epochs 30 --pseudos 25 --lsdim 64 --save weights/exp5/w6.h5; 
python VtPVAE.py  --epochs 30 --pseudos 25 --lsdim 128 --save weights/exp5/w7.h5; 