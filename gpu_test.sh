#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=gpu_test.out
#SBATCH --error=gpu_test.err
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

module purge
module load USS/2022
module load gcc/9.4.0-pe5.34
module load miniconda3/4.12.0
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0

conda activate ba_pointnet

echo "=== GPU ==="
nvidia-smi

echo "=== PyTorch ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU verfügbar:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Keine GPU')"