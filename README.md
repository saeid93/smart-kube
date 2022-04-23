# 1. Introduction
## 1.1. Repo contetns
This repo contains the code of the upcoming vpa aware scheduling paper

## Setup the environment in your machine
1. Download source code from GitHub
   ```
    git clone https://github.com/saeid93/CCGrid-paper
   ```
2. Download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
   ```
    conda create --name mobilekube python=3
   ```
4. Activate conda environment
   ```
    conda activate mobilekube
   ```
5. if you want to use GPUs make sure that you have the correct version of CUDA and cuDNN installed from [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
6. Use [PyTorch](https://pytorch.org/) or [Tensorflow](https://www.tensorflow.org/install/pip#virtual-environment-install) isntallation manual to install one of them based-on your preference

7. Install the followings
   ```
    sudo apt install cmake libz-dev
   ```
8. Install requirements
   ```
    pip install -r requirements.txt
   ```
9. setup [tensorboard monitoring](docs/monitorings/tensorboard.md)

# 2. Kubernetes Cluster Setup
If you want to do the real world Kubernetes experiemnts of the paper you should also do the following steps.
There are several options for setting up a Kuberentes cluster. The repo codes can connect to the cluster through the [Python client API](https://github.com/kubernetes-client/python) as long as you have access to the kube config address e.g. `~/.kube/config` in your config files specified.

We have used [Google Cloud Platform](https://cloud.google.com/) for our experiments.
You can find the toturial for creating the cluster on google cloud and locally in the followings:

* [Google cloud installation](docs/kubernetes/installation-gcp.md)
* [Local cluster setup installation](docs/kubernetes/installation-local.md)

# 3. Project Structure

1. [data](data)
2. [docs](docs)
3. [experiments](experiments)
4. [mobile-kube](mobile-kube)
5. [mobility dataset preprossing](mobility-dataset-preprossing)

The code is separated into three modules
   1. [data](data): This is the folder containing all the configs and results of the project. Could be anywhere in the project.
   3. [mobile-kube](mobile-kube): the core simulation library with [Open-AI gym](https://gym.openai.com/) interface
   4. [experiments](experiments): experiments of the paper and the reinforcement learning side of codes.

## 3.1. [mobile-kube](mobile-kube)
### Structure
* [src](mobile-kube/src): The folder containing the mobile-kube simulators. This should be installed for using.

### Usage
Go to the [mobile-kube](mobile-kube/mobile-kube) and install the library in the editable mode with
   ```
   pip install -e .
   ```

## 3.2. [data](data)
### Structure
Link the data folder (could be placed anywhere in your harddisk) to the project. A sample of the data folder is available at [data](data).

### Usage
Go to [experiments/utils/constants.py](experiments/utils/constants.py) and set the path to your data and project folders in the file. For example:
   ```
   DATA_PATH = "/Users/saeid/Codes/mobile-kube/data"
   ```
## 3.3. [mobility dataset preprossing](mobility-dataset-preprossing)
### Structure
* [ETL.py](mobility-dataset-preprossing/ETL.py)
* [README.md](mobility-dataset-preprossing/README.md)
* [Utils.py](mobility-dataset-preprossing/Utils.py)
* [main.py](mobility-dataset-preprossing/main.py)
* [requirements.txt](mobility-dataset-preprossing/requirements.txt)

### Usage
1. Install the requirements:
```bash
pip install -r requirements.txt
```
2. Run the code:
```bash
python main.py
```

3. The options are as follows:
```
Usage: main.py [OPTIONS]

Options:
  -d, --dataset TEXT      Directory of Cabspotting data set  [default:
                          data/*.txt]

  -g, --get BOOLEAN       Get data set from the internet  [default: False]
  -u, --url TEXT          The url of Cabspotting data set  [default: ]
  -i, --interval INTEGER  Enter the intervals between two points in seconds
                          [default: 100]

  --help                  Show this message and exit.

```
4. Enjoy!


formant of the parsed users mobility dataset.
Users data attributes:
1. Creating a CSV file with the size of: 2 * #NUM_OF_USERS + 1, each 3 column contains
the information of one user.
2. For instance, in case when there are 3 users in the network, we have a 7-column table as
follows:

| timesteps | lan1 | lat1 | lan2 | lat2 |
|-----------|------|------|------|------|
|0|35.76727803828804|51.35991084161459|35.76031345070641|1.39458643788907|
|1|35.76727803828856|51.35991084161443|35.76031345070765|1.39458643788633|
|2|35.76727803828856|51.35991084161443|35.76031345070765|1.39458643788633|

Copy the results at the data/dataset_metadata in a numbered foldered to use it in experiments generators.

## 3.4. [experiments](experiments)

### 3.4.1. [Data Generation](experiments/dataset)
The dataset, workloads, networks and traces are generated in the following order:
1. **Datasets**: Nodes, services, their capacities, requested resources and their initial placements. 
2. **Workloads**: The workload for each dataset that determines the resource usage at each time step. This is built on top of the datasets built on step 1. Each dataset can have several workloads.
3. **Networks**: The edge network for each dataset that generates the network with users and stations for the simulation. This contains the network object containing the nodes and stations and intitial location of the users. The network can be built both randomly of based-on the [Cabspotting](https://privamov.github.io/accio/docs/datasets.html) and [California towers stations](antennasearch.com) dataset. This is built on top of the nodes-services datasets built on step 1. Each dataset can have several networks.
4. **Traces**: The movement traces for each network. This is the location of each user at each timestep. This can be a random or based-on [Cabspotting](https://privamov.github.io/accio/docs/) dataset. This is built on top of the networks built in step 3.


To generate the datasets, workloads, networks and traces, first go to your data folder (remember data could be anywhere in your disk just point the data folder as  [experiments/utils/constants.py](experiments/utils/constants.py)).

#### 3.4.1.1. [Generating the Datasets](experiments/dataset/generate_dataset.py)

Go to the your dataset generation config [data/configs/dataset-generation/](data/configs/dataset-generation) make a folder named after your config and make the `config.json` in the folder e.g. see the `my-dataset` in the sample [data](data) folder [data/configs/generation-configs/dataset-generation/my-dataset/config.json](data/configs/generation-configs/dataset-generation/my-dataset/config.json). Then run the [experiments/dataset/generate_dataset.py](experiments/dataset/generate_dataset.py) with the following script:
```
python generate_dataset.py [OPTIONS]

Options:
  --dataset-config-folder TEXT      config-folder
  [default:                         my-dataset] 
```
For a full list of `config.json` parameters options see [dataset-configs-options](docs/configs-parameters/dataset-generation.md). The results will be saved in [data/datasets/<dataset_id>](data/datasets).

#### 4.4.1.2. [Generating the Workloads](experiments/dataset/generate_workload.py)

Go to the your workload generation config [data/configs/generation-configs/workload-generation](data/configs/generation-configs/dataset-generation) make a folder named after your config and make the `config.json` in the folder e.g. see the `my-workload` in the sample [data](data) folder [data/configs/generation-configs/workload-generation/my-workload/config.json](data/configs/generation-configs/dataset-generation/my-workload/config.json). For a full list of `config.json` see. Then run the [experiments/dataset/generate_dataset.py](experiments/dataset/generate_dataset.py) with the following script:
```
python generate_workload.py [OPTIONS]

Options:
  --workload-config-folder TEXT      config-folder
  [default:                          my-workload] 
```
For a full list of `config.json` parameters options see [workload-configs-options](docs/configs-parameters/workload-generation.md). The results will be saved in [data/datasets/<dataset_id>/<workload_id>](data/datasets).
<br />
### 4.4.1.3. [Generating the Networks](experiments/dataset/generate_network.py)

Go to the your dataset generation config [data/configs/generation-configs/network-generation](data/configs/generation-configs/network-generation) make a folder named after your config and make the `config.json` in the folder e.g. see the `my-network` in the sample [data](/data) folder [data/configs/generation-configs/dataset-generation/my-network/config.json](data/configs/generation-configs/dataset-generation/my-dataset/config.json). Then run the [experiments/dataset/generate_network.py](experiments/dataset/generate_network.py) with the following script:
```
python generate_network.py [OPTIONS]

Options:
  --network-config-folder TEXT      config-folder
  [default:                         my-network] 
```
For a full list of `config.json` parameters options see [network-configs-options](docs/configs-parameters/network-generation.md). The results will be saved in [data/datasets/<dataset_id>/<network_id>](data/datasets).

#### 4.4.1.4. [Generating the Traces](experiments/dataset/generate_trace.py)

Go to the your trace generation config [data/configs/generation-configs/trace-generation](data/configs/generation-configs/network-generation) make a folder named after your config and make the `config.json` in the folder e.g. see the `my-trace` in the sample [data](data) folder [data/configs/generation-configs/trace-generation/my-trace/config.json](data/configs/generation-configs/dataset-generation/my-dataset/config.json). Then run the [experiments/dataset/generate_trace.py](experiments/dataset/generate_trace.py) with the following script:
```
python generate_trace.py [OPTIONS]

Options:
  --trace-config-folder TEXT        config-folder
  [default:                         my-trace] 
```
For a full list of `config.json` parameters options see [trace-configs-options](docs/configs-parameters/trace-generation.md). The results will be saved in [data/datasets/<dataset_id>/<network_id>/<trace_id>](data/datasets).

### 4.4.2. [Training](experiments/training) and [analysis](experiments/analysis)

#### 4.4.2.1. [Training the agent](experiments/training/learner.py)

1. change the training parameters in `<configs-path>/real/<experiment-folder>/config_run.json`. For more information about the hyperparamters in this json file see [hyperparameter guide](docs/learning/hyperparameter-guide.md)
2. To train the environments go to the parent folder and run the following command.
```
python experiments/learning/learners.py --mode real --local-mode false --config-folder PPO --type-env 0 --dataset-id 0 --workload-id 0 --use-callback true
```

### 4.4.2. [Analysis](experiments/analysis)

#### 4.4.2.1 [check_env](experiments/check_env.py)

#### 4.4.2.2 [check_learned](experiments/check_learned.py)

#### 4.4.2.3 [test_baselines](experiments/test_baselines.py)

#### 4.4.2.4 [check environment](experiments/test_rls.py)

#### 4.4.2.5 [check environment](experiments/analysis_test.ipynb)

#### 4.4.2.6 [check environment](experiments/analysis_train.ipynb)


### 4.4.4. [Kubernetes interface](mobile-kube/src/smart_scheduler/util/kubernetes_utils)

The Kubernetes interface is designed based-on the Kubernetes [api version 1](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.21/).

The main operations that are currently implemented are:
* creating
  * cluster
  * utilisation server
  * pods
* actions
  * moving pods
  * deleting pods
  * cleaning nemespace
* monitoring
  * get nodes resource usage
  * get pods resource usage

a sample of using the interface can be found [here](experiments/kube_operations.py)

# 4. Sample run on GKE cluster

Log of a running emulation - moving service 0 from node 1 to node 0 (s0n1 -> s0n1)

![logs](docs/images/logs.png)

Google cloud console of a running emulation - moving service 0 from node 1 to node 0 (s0n1 -> s0n1)

![images](docs/images/gcloud-console.png)

# 5. Other

1. [Step by step guide to trainig the code on EECS](docs/step-by-step-guides/EECS.md)

2. [Step by step guide to trainig the code on GKE](docs/step-by-step-guides/GKE.md)

3. [List of running ray problems](docs/problems/ray.md)

4. [List of QMUL EECS problems](docs/problems/eecs-server-setup.md)

5. [Tensorboard Monitoring](docs/monitorings/tensorboard.md)

6. [Cluster Monitoring](docs/monitorings/cluster.md)
