networks=(1 5 6)
for network in ${networks[@]}
do
   /homes/sg324/.conda/envs/central/bin/python \
   /homes/sg324/mobile-kube/experiments/training/train.py \
   --series $1 --config-file final-$2 --cluster-id 6 --network-id $network --trace-id 2
done