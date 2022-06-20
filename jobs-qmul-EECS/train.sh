module load anaconda3
source activate central
cd ~/smart-scheduler
/homes/sg324/.conda/envs/central/bin/python \
/homes/sg324/smart-scheduler/experiments/training/train.py \
--series $1 --config-file $2 --cluster-id $3 --workload-id $4