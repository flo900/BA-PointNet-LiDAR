#!/bin/bash

module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0

conda activate ba_pointnet

echo "BA environment ready"
echo "Python: $(which python)"
python --version