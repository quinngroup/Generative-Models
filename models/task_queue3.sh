#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python VtPVAE.py  --epochs 50 --lsdim 16 --save weights/exp15/lr1.h5 --log obs8/lr1 --lr 1e-1;
python VtPVAE.py  --epochs 50 --lsdim 16 --save weights/exp15/lr2.h5 --log obs8/lr2 --lr 1e-2;
python VtPVAE.py  --epochs 50 --lsdim 16 --save weights/exp15/lr3.h5 --log obs8/lr3 --lr 1e-3;
python VtPVAE.py  --epochs 50 --lsdim 16 --save weights/exp15/lr4.h5 --log obs8/lr4 --lr 1e-4;
python VtPVAE.py  --epochs 50 --lsdim 16 --save weights/exp15/lr5.h5 --log obs8/lr5 --lr 1e-5;
python VtPVAE.py  --epochs 50 --lsdim 16 --save weights/exp15/lr6.h5 --log obs8/lr6 --lr 1e-6;
