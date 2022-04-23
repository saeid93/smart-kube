algs=("PG" "A2C" "A3C" "PPO" "IMPALA")
for alg in ${algs[@]}
do
   /homes/sg324/.conda/envs/central/bin/python \
   /homes/sg324/mobile-kube/experiments/training/train.py \
   --series $1 --config-file final-$alg --dataset-id $2 --network-id $3 --trace-id $4
done