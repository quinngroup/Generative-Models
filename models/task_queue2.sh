#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python 3DConv.py --epochs 30 --lsdim 16 --load weights/3dconv/b1.h5 --pseudos 25 --batch-size 64 --save weights/3dconv/b1e2.h5;

python 3DConv.py --epochs 30 --lsdim 16 --load weights/3dconv/b2.h5 --pseudos 25 --batch-size 128 --save weights/3dconv/b2e2.h5

python 3DConv.py --epochs 30 --lsdim 16 --save weights/3dconv/b3.h5 --pseudos 25 --batch-size 200;

python 3DConv.py --epochs 30 --lsdim 16 --load weights/3dconv/b3.h5 --pseudos 25 --batch-size 200 --save weights/3dconv/b3e2.h5;

python 3Dconv.py --epochs 30 --lsdim 16 --load weights/3dconv/b1e2.h5 --pseudos 25 --batch-size 64 --save weights/3dconv/b1e3.h5

python 3Dconv.py --epochs 30 --lsdim 16 --load weights/3dconv/b2e2.h5 --pseudos 25 --batch-size 128 --save weights/3dconv/b2e3.h5

python 3Dconv.py --epochs 30 --lsdim 16 --load weights/3dconv/b3e2.h5 --pseudos 25 --batch-size 200 --save weights/3dconv/b3e3.h5
