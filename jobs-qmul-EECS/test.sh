/homes/sg324/.conda/envs/central/bin/python \
/homes/sg324/smart-scheduler/experiments/analysis/test_rls.py \
--train-series $1 --test-series $1 --workload-id 1 --experiment-id $j \
--trace-id-test $i --checkpoint-to-load $checkpoint