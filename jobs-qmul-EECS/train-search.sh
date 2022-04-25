for i in {1..3}
do
   /homes/sg324/.conda/envs/central/bin/python /homes/sg324/mobile-kube/experiments/training/train.py --series $1 --config-file PPO-search-$i --cluster-id 6 --network-id $2 --trace-id 2
done
