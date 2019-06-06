#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/p5.h5 --pseudos 5; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/p15.h5 --pseudos 15; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/p25.h5 --pseudos 25; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/p35.h5 --pseudos 35; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/p45.h5 --pseudos 45; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/p55.h5 --pseudos 55; 
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/p65.h5 --pseudos 65;
python VtPVAE.py  --lsdim 12 --gamma .15 --save weights/exp3/p75.h5 --pseudos 75;
