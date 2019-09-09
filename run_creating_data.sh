#!/bin/bash
mode=2 # 0: direct, 1: indirect (loading graphs), indirect_hops, 3: generic

if [ ${mode} == 0 ]; then
  python create_geometric_direct.py \
    --in_dataset AGGF1 \
    --out_dataset AGGF1 \
    --in_folder /home/dinh/data/projects/code/pycharm/graphprot2/data/raw_cv \
    --out_folder /home/dinh/data/projects/code/pycharm/graphprot2/data/geometric_cv
elif [ ${mode} == 1  ]; then
  python create_geometric_indirect.py \
    --in_dataset AGGF1 \
    --out_dataset AGGF1 \
    --in_folder /home/dinh/data/projects/code/pycharm/graphprot2/data/raw_cv \
    --out_folder /home/dinh/data/projects/code/pycharm/graphprot2/data/geometric_cv
elif [ ${mode} == 2  ]; then
  python create_geometric_indirect_hops.py \
    --in_dataset AGGF1 \
    --out_dataset AGGF1 \
    --in_folder /home/dinh/data/projects/code/pycharm/graphprot2/data/raw_cv \
    --out_folder /home/dinh/data/projects/code/pycharm/graphprot2/data/geometric_cv    
else
    python create_geometric_generic.py \
    --in_dataset GENERIC \
    --out_dataset GENERIC \
    --in_folder /home/dinh/data/projects/code/pycharm/graphprot2/data/raw_cv \
    --out_folder /home/dinh/data/projects/code/pycharm/graphprot2/data/geometric_cv
fi
