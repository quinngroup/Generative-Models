#!/bin/bash

dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

python VtPVAE.py --save runs/exp17/plr2e-7.h5 --log runs/exp17/plr2e-7 --plr 2e-7
python VtPVAE.py --save runs/exp17/plr4e-7.h5 --log runs/exp17/plr4e-7 --plr 4e-7
python VtPVAE.py --save runs/exp17/plr6e-7.h5 --log runs/exp17/plr6e-7 --plr 6e-7
python VtPVAE.py --save runs/exp17/plr2e-6.h5 --log runs/exp17/plr2e-6 --plr 2e-6
python VtPVAE.py --save runs/exp17/plr4e-6.h5 --log runs/exp17/plr4e-6 --plr 4e-6
python VtPVAE.py --save runs/exp17/plr6e-6.h5 --log runs/exp17/plr6e-6 --plr 6e-6
python VtPVAE.py --save runs/exp17/plr2e-5.h5 --log runs/exp17/plr2e-5 --plr 2e-5
python VtPVAE.py --save runs/exp17/plr4e-5.h5 --log runs/exp17/plr4e-5 --plr 4e-5
python VtPVAE.py --save runs/exp17/plr6e-5.h5 --log runs/exp17/plr6e-5 --plr 6e-5