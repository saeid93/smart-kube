for i in {0..4}
do
   /homes/sg324/.conda/envs/central/bin/python \
   /homes/sg324/mobile-kube/experiments/training/train.py \
   --series $1 --config-file $2-$i --dataset-id $3 --network-id $4 --trace-id $5
done