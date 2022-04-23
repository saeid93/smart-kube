# The loop for trainnig RL methods
for i in {0..4}
    do
    /homes/sg324/.conda/envs/central/bin/python \
    /homes/sg324/mobile-kube/experiments/training/train.py \
    --series $1 --config-file PPO-$i --dataset-id 6 --network-id $2 --trace-id 2
    done

# The loop for testing RL methods
for i in {0,2}
    do
    for j in {0..4}
        do
            /homes/sg324/.conda/envs/central/bin/python \
            /homes/sg324/mobile-kube/experiments/analysis/test_rls.py \
            --train-series $1 --test-series $1 --network-id $2 --experiment-id $j \
            --trace-id 2 --trace-id-test $i --episode-length 10
        done
    done

# The loop for testing the baselines
for i in {0,2}
    do
    for j in {"sim-binpacking","sim-greedy"}
        do
        /homes/sg324/.conda/envs/central/bin/python \
        /homes/sg324/mobile-kube/experiments/analysis/test_baselines.py \
        --comp-train-series $1 --test-series $1 --network-id $2 --comp-experiment-id 0 \
        --trace-id 2 --trace-id-test $i --type-env $j --episode-length 10
        done
    done