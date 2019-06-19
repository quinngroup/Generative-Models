#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)


python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 7e-5 --log obs5/pl7e-5 --save weights/exp12/pl7e-5.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 3e-5 --log obs5/pl3e-5 --save weights/exp12/pl3e-5.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 1e-5 --log obs5/pl1e-5 --save weights/exp12/pl1e-5.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 7e-6 --log obs5/pl7e-6 --save weights/exp12/pl7e-6.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 3e-6 --log obs5/pl3e-6 --save weights/exp12/pl3e-6.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 1e-6 --log obs5/pl1e-6 --save weights/exp12/pl1e-6.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 7e-7 --log obs5/pl7e-7 --save weights/exp12/pl7e-7.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 3e-7 --log obs5/pl3e-7 --save weights/exp12/pl3e-7.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 1e-7 --log obs5/pl1e-7 --save weights/exp12/pl1e-7.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 7e-8 --log obs5/pl7e-8 --save weights/exp12/pl7e-8.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 3e-8 --log obs5/pl3e-8 --save weights/exp12/pl3e-8.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 1e-8 --log obs5/pl1e-8 --save weights/exp12/pl1e-8.h5;


