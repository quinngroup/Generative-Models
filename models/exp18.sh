#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python VtPVAE.py --save runs/exp18/unnormed30.h5 --log runs/exp18/unnormed30 --epochs 30
python VtPVAE.py --save runs/exp18/unnormed60.h5 --log runs/exp18/unnormed60 --epochs 60
python VtPVAE.py --save runs/exp18/unnormed90.h5 --log runs/exp18/unnormed90 --epochs 90
python VtPVAE.py --save runs/exp18/notracking30.h5 --log runs/exp18/notracking30 --epochs 30 --batchNorm
python VtPVAE.py --save runs/exp18/notracking60.h5 --log runs/exp18/notracking60 --epochs 60 --batchNorm
python VtPVAE.py --save runs/exp18/notracking90.h5 --log runs/exp18/notracking90 --epochs 90 --batchNorm
python VtPVAE.py --save runs/exp18/running30.h5 --log runs/exp18/running30 --epochs 30 --batchNorm --batch-tracking
python VtPVAE.py --save runs/exp18/running60.h5 --log runs/exp18/running60 --epochs 60 --batchNorm --batch-tracking
python VtPVAE.py --save runs/exp18/running90.h5 --log runs/exp18/running90 --epochs 90 --batchNorm --batch-tracking