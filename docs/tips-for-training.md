1. Start with a very small dataset e.g. 2 servers and 10 arriving containers
2. Use the check_env script and try scale the rewards within some range
3. Train the model using bash script files with gridsearch (it is RL and it's a huge pain to train) and monitor the training process in tensorboard
4. If some promissing situations in the training then add that to some promissing results for test
5. For testing use the real arabesque workloads and test on them
6. GOTO step 1 and make the dataset bigger (this should be subjected to the Arabesque dataset size and GKE cluster node sizes as we will finally deploy on them)
7. Repeat 1-6 until some good dataset with all the tests in good shape
8. If all good run tests on K8s