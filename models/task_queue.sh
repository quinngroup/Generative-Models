#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/1x5x15.h5 --pseudos 5 --batch-size 1000; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/1x25x15.h5 --pseudos 25 --batch-size 1000; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/1x45x15.h5 --pseudos 45 --batch-size 1000; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/1x65x15.h5 --pseudos 65 --batch-size 1000; 
python VtPVAE.py  --lsdim 12 --gamma .05 --save weights/exp3/0x5x05.h5 --pseudos 5; 
python VtPVAE.py  --lsdim 12 --gamma .05 --save weights/exp3/0x25x05.h5 --pseudos 25; 
python VtPVAE.py  --lsdim 12 --gamma .05 --save weights/exp3/0x45x05.h5 --pseudos 45;
python VtPVAE.py  --lsdim 12 --gamma .05 --save weights/exp3/0x65x05.h5 --pseudos 65;
