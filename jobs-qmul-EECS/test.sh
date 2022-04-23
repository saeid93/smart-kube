# The loop for testing RL methods
checkpoints=(001000)
for checkpoint in ${checkpoints[@]}
    do
    for i in {0,2}
        do
        for j in {0..4}
            do
                /homes/sg324/.conda/envs/central/bin/python \
                /homes/sg324/mobile-kube/experiments/analysis/test_rls.py \
                --train-series $1 --test-series $1 --network-id 1 --experiment-id $j \
                --trace-id 2 --trace-id-test $i --checkpoint-to-load $checkpoint
            done
        done
    done