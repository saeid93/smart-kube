module load cuda/11.2-cudnn8.1.0
module load anaconda3
source activate central
/homes/sg324/.conda/envs/central/bin/python experiments/training/train.py --series 36 --dataset-id 6 --network-id 1 --trace-id 2