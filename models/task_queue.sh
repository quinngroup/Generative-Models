#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/1x5x15.h5 --pseudos 5 --batch-size 1000; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/1x15x15.h5 --pseudos 15 --batch-size 1000; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/1x25x15.h5 --pseudos 25 --batch-size 1000; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/1x35x15.h5 --pseudos 35 --batch-size 1000; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/1x45x15.h5 --pseudos 45 --batch-size 1000; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/1x55x15.h5 --pseudos 55 --batch-size 1000; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/1x65x15.h5 --pseudos 65 --batch-size 1000;
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/1x75x15.h5 --pseudos 75 --batch-size 1000;
