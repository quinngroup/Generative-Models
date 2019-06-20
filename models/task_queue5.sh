#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)


python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 7e-5 --log obs5/pl7e5 --save weights/exp12/pl7e5.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 4e-5 --log obs5/pl4e5 --save weights/exp12/pl4e5.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 1e-5 --log obs5/pl1e5 --save weights/exp12/pl1e5.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 7e-6 --log obs5/pl7e6 --save weights/exp12/pl7e6.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 4e-6 --log obs5/pl4e6 --save weights/exp12/pl4e6.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 1e-6 --log obs5/pl1e6 --save weights/exp12/pl1e6.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 7e-7 --log obs5/pl7e7 --save weights/exp12/pl7e7.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 4e-7 --log obs5/pl4e7 --save weights/exp12/pl4e7.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 1e-7 --log obs5/pl1e7 --save weights/exp12/pl1e7.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 7e-8 --log obs5/pl7e8 --save weights/exp12/pl7e8.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 4e-8 --log obs5/pl4e8 --save weights/exp12/pl4e8.h5;
python 3DConv.py --epochs 30 --lsdim 16 --pseudos 25 --batch-size 50 --lr 1e-3 --plr 1e-8 --log obs5/pl1e8 --save weights/exp12/pl1e8.h5;


