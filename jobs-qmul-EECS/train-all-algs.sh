for i in {0..4}
do
   echo /homes/sg324/.conda/envs/central/bin/python /homes/sg324/mobile-kube/experiments/training/train.py --series $1 --config-file PG-$i --dataset-id 6 --network-id $2 --trace-id 2
done

for i in {0..4}
do
   echo /homes/sg324/.conda/envs/central/bin/python /homes/sg324/mobile-kube/experiments/training/train.py --series $1 --config-file PPO-$i --dataset-id 6 --network-id $2 --trace-id 2
done

for i in {0..4}
do
   echo /homes/sg324/.conda/envs/central/bin/python /homes/sg324/mobile-kube/experiments/training/train.py --series $1 --config-file A2C-$i --dataset-id 6 --network-id $2 --trace-id 2
done

for i in {0..4}
do
   echo /homes/sg324/.conda/envs/central/bin/python /homes/sg324/mobile-kube/experiments/training/train.py --series $1 --config-file A3C-$i --dataset-id 6 --network-id $2 --trace-id 2
done

for i in {0..4}
do
   echo /homes/sg324/.conda/envs/central/bin/python /homes/sg324/mobile-kube/experiments/training/train.py --series $1 --config-file IMPALA-$i --dataset-id 6 --network-id $2 --trace-id 2
done
