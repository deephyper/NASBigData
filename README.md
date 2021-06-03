# AgEBO-Tabular

[![DOI](https://zenodo.org/badge/279793726.svg)](https://zenodo.org/badge/latestdoi/279793726)

Aging Evolution with Bayesian Optimization (AgEBO) is a nested-distributed algorithm to generate better neural architectures. AgEBO advantages are:

- the parallel evaluation of neural networks on computing ressources (e.g., cores, gpu, nodes).
- the parallel training of each evaluated neural networks by using data-parallelism (Horovod).
- the jointly optimization of hyperparameters and neural architectures which enables the automatic adaptation of data-parallelism setting to avoid a loss of accuracy.

This repo contains the experimental materials linked to the implementation of AgEBO algorithm in DeepHyper's repo.
The version of DeepHyper used is: [e8e07e2db54dceed83b626104b66a07509a95a8c](https://github.com/deephyper/deephyper/commit/e8e07e2db54dceed83b626104b66a07509a95a8c)

## Environment information

The experiments were executed on the [ThetaGPU](https://www.alcf.anl.gov/alcf-resources/theta) supercomputer.

* OS Login Node: Ubuntu 18.04.5 LTS (GNU/Linux 4.15.0-112-generic x86_64)
* OS Compute Node: NVIDIA DGX Server Version 4.99.9 (GNU/Linux 5.3.0-62-generic x86_64)
* Python: Miniconda Python 3.8

For more information about the environment refer to the `infos-sc21.txt` which was generated with the provided SC [Author-Kit](https://github.com/SC-Tech-Program/Author-Kit.)

## Installation

Install Miniconda: [conda.io](https://docs.conda.io/en/latest/miniconda.html). Then create a Python environment:

```console
conda create -n dh-env python=3.8

```

Then install Deephyper. To have the detailed installation process of DeepHyper follow the instructions given at: [deephyper.readthedocs.io](https://deephyper.readthedocs.io/). We propose the following commands:

```console
conda activate dh-env
conda install gxx_linux-64 gcc_linux-64 -y
git clone https://github.com/deephyper/deephyper.git
cd deephyper/
git checkout e8e07e2db54dceed83b626104b66a07509a95a8c
pip install -e.
pip install ray[default]
```

Finally, install the NASBigData package::

```console
cd ..
git clone https://github.com/deephyper/NASBigData.git
cd NASBigData/
pip install -e.
```

## Download and Generate datasets from ECP-Candle

Have the following dependencies installed:

```console
pip install numba
pip install astropy
pip install patsy
pip install statsmodels
```

For the Combo dataset run:

```console
cd NASBigData/nas_big_data/combo/
sh download_data.sh
```

For the Attn dataset run:

```console
cd NASBigData/nas_big_data/attn/
sh download_data.sh
```

## How it works

The AgEBO algorithm (Aging Evolution with Bayesian Optimisation) was directly added to the DeepHyper project and can be found [here](https://github.com/deephyper/deephyper/blob/e8e07e2db54dceed83b626104b66a07509a95a8c/deephyper/search/nas/agebo.py#L90).

To submit and run an experiment on the ThetaGPU system the following command is used:

```console
deephyper ray-submit nas agebo -w combo_2gpu_8_agebo_sync -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 2 --num-gpus-per-task 2 -as ../SetUpEnv.sh --n-jobs 16
```

where

* `-w` denotes the name of the experiment.
* `-n` denotes the number of nodes requested.
* `-t` denotes the allocation time (minutes) requested.
* `-A` denotes the project's name at the ALCF.
* `-q` denotes the queue's name.
* `--problem` is the Python package import to the Problem definition (which define the hyperparameter and neural architecture search space, the loss to optimise, etc.).
* `--run` is the Python package import to the run function (which evaluate each configuration sampled by the search).
* `--max-evals` denotes the maximum number of evaluations to performe (often affected to an high value so that the search uses the whole allocation time).
* `--num-cpus-per-task` the number of cores used by each evaluation.
* `--num-gpus-per-task` the number of GPUs used by each evaluation.
* `--as` the absolute PATH to the activation script `SetUpEnv.sh` (used to initialise the good environment on compute nodes when the allocation is starting).
* `--n-jobs` the number of processes that the surrogate model of the Bayesian optimiser can use.


The `deephyper ray-submit ...` command will create a directory with `-w` name and automatically generate a submission script for Cobalt (the scheduler at the ALCF). Such a submission script will be composed of the following.

The initialisation of the environment:

```bash
#!/bin/bash -x
#COBALT -A datascience
#COBALT -n 8
#COBALT -q full-node
#COBALT -t 180

mkdir infos && cd infos

ACTIVATE_PYTHON_ENV="/lus/grand/projects/datascience/regele/thetagpu/agebo/SetUpEnv.sh"
echo "Script to activate Python env: $ACTIVATE_PYTHON_ENV"
source $ACTIVATE_PYTHON_ENV

```

The initialisation of the Ray cluster:

```bash
# USER CONFIGURATION
CPUS_PER_NODE=8
GPUS_PER_NODE=8


# Script to launch Ray cluster
# Getting the node names
mapfile -t nodes_array -d '\n' < $COBALT_NODEFILE

head_node=${nodes_array[0]}
head_node_ip=$(dig $head_node a +short | awk 'FNR==2')

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# Starting the Ray Head Node
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
ssh -tt $head_node_ip "source $ACTIVATE_PYTHON_ENV; \
    ray start --head --node-ip-address=$head_node_ip --port=$port \
    --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE --block" &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((${#nodes_array[*]} - 1))
echo "$worker_num workers"

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    node_i_ip=$(dig $node_i a +short | awk 'FNR==1')
    echo "Starting WORKER $i at $node_i with ip=$node_i_ip"
    ssh -tt $node_i_ip "source $ACTIVATE_PYTHON_ENV; \
        ray start --address $ip_head \
        --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE" --block &
    sleep 5
done

```

The DeepHyper command to start the search:

```bash
deephyper nas agebo --evaluator ray --ray-address auto \
    --problem nas_big_data.combo.problem_agebo.Problem \
    --run deephyper.nas.run.tf_distributed.run \
    --max-evals 10000 \
    --num-cpus-per-task 2 \
    --num-gpus-per-task 2 \
    --n-jobs=16
```

## Commands to reproduce

The experiments are name as `{dataset}_{x}gpu_{y}_{z}_{other}` where

* `dataset` is the name of the corresponding dataset (e.g., combo or attn).
* `x` is the number of GPUs used for each trained neural network (e.g., 1, 2, 4, 8).
* `y` is the number of nodes used for the allocation (e.g., 1, 2, 4, 8, 16).
* `z` is the name of the algorithm (e.g., age, agebo).
* `other` are other keywords used to differentiate some experiments (e.g., kappa value)>

We give the full set of commands used to run our experiments.

### Combo dataset

* combo_1gpu_8_age

```console
deephyper ray-submit nas regevo -w combo_1gpu_8_age -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_ae.Problem --run deephyper.nas.run.alpha.run --max-evals 10000 --num-cpus-per-task 1 --num-gpus-per-task 1 -as ../SetUpEnv.sh
```

* combo_2gpu_8_age

```console
deephyper ray-submit nas regevo -w combo_2gpu_8_age -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_ae.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 2 --num-gpus-per-task 2 -as ../SetUpEnv.sh
```

* combo_8gpu_8_age

```console
deephyper ray-submit nas regevo -w combo_8gpu_8_age -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_ae.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 8 --num-gpus-per-task 8 -as ../SetUpEnv.sh
```

* combo_8gpu_8_agebo

```console
deephyper ray-submit nas agebo -w combo_8gpu_8_agebo -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 8 --num-gpus-per-task 8 -as ../SetUpEnv.sh --n-jobs 16
```

* combo_2gpu_8_agebo

```console
deephyper ray-submit nas agebo -w combo_2gpu_8_agebo -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 2 --num-gpus-per-task 2 -as ../SetUpEnv.sh --n-jobs 16
```

* combo_1gpu_2_age

```console
deephyper ray-submit nas regevo -w combo_1gpu_2_age -n 2 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_ae.Problem --run deephyper.nas.run.alpha.run --max-evals 10000 --num-cpus-per-task 1 --num-gpus-per-task 1 -as ../SetUpEnv.sh
```

* combo_2gpu_4_age

```console
deephyper ray-submit nas regevo -w combo_2gpu_4_age -n 4 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_ae.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 2 --num-gpus-per-task 2 -as ../SetUpEnv.sh
```

* combo_4gpu_8_age

```console
deephyper ray-submit nas regevo -w combo_4gpu_8_age -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_ae.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 4 --num-gpus-per-task 4 -as ../SetUpEnv.sh
```

* combo_8gpu_16_age

```console
deephyper ray-submit nas regevo -w combo_8gpu_16_age -n 16 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_ae.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 8 --num-gpus-per-task 8 -as ../SetUpEnv.sh
```

* combo_1gpu_2_agebo

```console
deephyper ray-submit nas agebo -w combo_1gpu_2_agebo -n 2 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.alpha.run --max-evals 10000 --num-cpus-per-task 1 --num-gpus-per-task 1 -as ../SetUpEnv.sh --n-jobs 16
```

* combo_2gpu_4_agebo

```console
deephyper ray-submit nas agebo -w combo_2gpu_4_agebo -n 4 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 2 --num-gpus-per-task 2 -as ../SetUpEnv.sh --n-jobs 16
```

* combo_4gpu_8_agebo

```console
deephyper ray-submit nas agebo -w combo_4gpu_8_agebo -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 4 --num-gpus-per-task 4 -as ../SetUpEnv.sh --n-jobs 16
```

* combo_8gpu_16_agebo

```console
deephyper ray-submit nas agebo -w combo_8gpu_16_agebo -n 16 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 8 --num-gpus-per-task 8 -as ../SetUpEnv.sh --n-jobs 16
```

* combo_4gpu_8_agebo_1_96

```console
deephyper ray-submit nas agebo -w combo_4gpu_8_agebo_1_96 -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 4 --num-gpus-per-task 4 -as ../SetUpEnv.sh --n-jobs 16 --kappa 1.96
```

* combo_4gpu_8_agebo_19_6

```console
deephyper ray-submit nas agebo -w combo_4gpu_8_agebo_19_6 -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 4 --num-gpus-per-task 4 -as ../SetUpEnv.sh --n-jobs 16 --kappa 19.6
```

* combo_1gpu_8_agebo

```console
deephyper ray-submit nas agebo -w combo_1gpu_8_agebo -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.alpha.run --max-evals 10000 --num-cpus-per-task 1 --num-gpus-per-task 1 -as ../SetUpEnv.sh --n-jobs 16
```

* combo_4gpu_8_ambsmixed

```console
deephyper ray-submit nas ambsmixed -w combo_4gpu_8_ambsmixed -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 4 --num-gpus-per-task 4 -as ../SetUpEnv.sh --n-jobs 16
```

* combo_4gpu_8_regevomixed

```console
deephyper ray-submit nas regevomixed -w combo_4gpu_8_regevomixed -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 4 --num-gpus-per-task 4 -as ../SetUpEnv.sh
```

* combo_2gpu_1_age

```console
deephyper ray-submit nas regevo -w combo_2gpu_1_age -n 1 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_ae.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 2 --num-gpus-per-task 2 -as ../SetUpEnv.sh
```

* combo_2gpu_2_age

```console
deephyper ray-submit nas regevo -w combo_2gpu_2_age -n 2 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_ae.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 2 --num-gpus-per-task 2 -as ../SetUpEnv.sh
```

* combo_2gpu_16_age

```console
deephyper ray-submit nas regevo -w combo_2gpu_16_age -n 16 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_ae.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 2 --num-gpus-per-task 2 -as ../SetUpEnv.sh
```

* combo_2gpu_1_agebo

```console
deephyper ray-submit nas agebo -w combo_2gpu_1_agebo -n 1 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 2 --num-gpus-per-task 2 -as ../SetUpEnv.sh --n-jobs 16
```

* combo_2gpu_2_agebo

```console
deephyper ray-submit nas agebo -w combo_2gpu_2_agebo -n 2 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 2 --num-gpus-per-task 2 -as ../SetUpEnv.sh --n-jobs 16
```

* combo_2gpu_16_agebo

```console
deephyper ray-submit nas agebo -w combo_2gpu_16_agebo -n 16 -t 180 -A datascience -q full-node --problem nas_big_data.combo.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 2 --num-gpus-per-task 2 -as ../SetUpEnv.sh --n-jobs 16
```

### Attn dataset

* attn_1gpu_8_age

```console
deephyper ray-submit nas regevo -w attn_1gpu_8_age -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.attn.problem_ae.Problem --run deephyper.nas.run.alpha.run --max-evals 10000 --num-cpus-per-task 1 --num-gpus-per-task 1 -as ../SetUpEnv.sh
```

* attn_1gpu_8_agebo

```console
deephyper ray-submit nas agebo -w attn_1gpu_8_agebo -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.attn.problem_agebo.Problem --run deephyper.nas.run.alpha.run --max-evals 10000 --num-cpus-per-task 1 --num-gpus-per-task 1 -as ../SetUpEnv.sh --n-jobs 16
```

* attn_2gpu_8_agebo

```console
deephyper ray-submit nas agebo -w attn_2gpu_8_agebo -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.attn.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 2 --num-gpus-per-task 2 -as ../SetUpEnv.sh --n-jobs 16
```

* attn_4gpu_8_agebo

```console
deephyper ray-submit nas agebo -w attn_4gpu_8_agebo -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.attn.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 4 --num-gpus-per-task 4 -as ../SetUpEnv.sh --n-jobs 16
```

* attn_8gpu_8_agebo

```console
deephyper ray-submit nas agebo -w attn_8gpu_8_agebo -n 8 -t 180 -A datascience -q full-node --problem nas_big_data.attn.problem_agebo.Problem --run deephyper.nas.run.tf_distributed.run --max-evals 10000 --num-cpus-per-task 8 --num-gpus-per-task 8 -as ../SetUpEnv.sh --n-jobs 16
```
