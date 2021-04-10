# AgEBO-Tabular

[![DOI](https://zenodo.org/badge/279793726.svg)](https://zenodo.org/badge/latestdoi/279793726)

Aging Evolution with Bayesian Optimization (AgEBO) is a nested-distributed algorithm to generate better neural architectures. AgEBO advantages are:

- the parallel evaluation of neural networks on computing ressources (e.g., cores, gpu, nodes).
- the parallel training of each evaluated neural networks by using data-parallelism (Horovod).
- the jointly optimization of hyperparameters and neural architectures which enables the automatic adaptation of data-parallelism setting to avoid a loss of accuracy.

This repo contains the experimental materials linked to the implementation of AgEBO algorithm in DeepHyper's repo.
The version of DeepHyper used is: [ad5a0f3391b6afca4358c66123246d0086c02f0f](https://github.com/deephyper/deephyper/commit/ad5a0f3391b6afca4358c66123246d0086c02f0f)

## Installation

To install DeepHyper follow the instructions given at: [deephyper.readthedocs.io](https://deephyper.readthedocs.io/)

After installing DeepHyper, install this repo by running the following commands in your terminal:

```bash
git clone https://github.com/deephyper/NASBigData.git
cd NASBigData/
pip install -e.
```

### Installation of DeepHyper on Cooley

On Cooley at ALCF:

Inside `~/.soft.cooley`:

```text
+mvapich2
@default
```

Then:

```bash
soft add +gcc-6.3.0
HOROVOD_WITH_TENSORFLOW=1 pip install horovod[tensorflow]
```

Having miniconda installed, execute:

```bash
cd project/datascience/
conda create -p dh-env python=3.7
```

Go on compute node, `qsub -I -A datascience -n 1 -t 30 -q debug`, then:

```bash
cd project/datascience/
conda activate dh-env/
conda install -c anaconda tensorflow=1.15
git clone https://github.com/deephyper/deephyper.git
cd deephyper/
pip install -e .
```

After intsalling DeepHyper:

```bash
pip install autogluon
conda install -c anaconda mxnet
```

For cuda: `pip install mxnet-cu101`

## Main commands to reproduce

## Balsam launcher

Before creating balsam-applications or jobs you need to create the balsam database and start it:

```bash
balsam init mydatabase
source balsamactivate mydatabase
```

### Aging Evolution for neural architecture search

Create the `AgE` application for Balsam on Theta:

```bash
balsam app --name AgE --exe "$(which python) -m deephyper.search.nas.regevo --evaluator balsam --run deephyper.nas.run.horovod.run"
```

Then create a job for a specific experiment (here models are evaluated with 8 ranks in parallel):

```bash
balsam job --name covertype_age_129 --workflow covertype_age_129 --app AgE --args "--problem nas_big_data.covertype.problem_ae.Problem --max-evals 1000 --num-threads-per-rank 16 --num-ranks-per-node 8"
```

Finally, launch this experiment (for Theta):

```bash
balsam submit-launch -n 129 -t 180 -A $project_name -q default --job-mode mpi --wf-filter covertype_age_129
```

### Aging Evolution with Bayesian Optimization (AgEBO)

Create the `AgEBO` application for Balsam on Theta:

```bash
balsam app --name AgEBO --exe "$(which python) -m deephyper.search.nas.agebov3 --evaluator balsam --run deephyper.nas.run.horovod.run"
```

Then create a job for a specific experiment (here models are evaluated with 8 ranks in parallel):

```bash
balsam job --name covertype_agebo_129 --workflow covertype_agebo_129 --app AgE --args "--problem nas_big_data.covertype.problem_agebov3.Problem --max-evals 1000 --num-threads-per-rank 16 --num-ranks-per-node 8"
```

Finally, launch this experiment (for Theta):

```bash
balsam submit-launch -n 129 -t 180 -A $project_name -q default --job-mode mpi --wf-filter covertype_agebo_129
```

### Aging Evolution for joint optimization

Aging Evolution to optimize both hyperparameters and neural architectures, local testing with dummy evaluation function:

```bash
python -m nas_big_data.search.ae_hpo_nas --run nas_big_data.run.quick.run --problem nas_big_data.covertype.problem_agebov4_skopt.Problem --max-evals 1000
```

Create the `AgEHPNAS` application for Balsam on Theta:

```bash
balsam app --name AgEHPNAS --exe "$(which python) -m nas_big_data.search.ae_hpo_nas --evaluator balsam --run deephyper.nas.run.horovod.run"
```

Then create a job for a specific experiment:

```bash
balsam job --name covertype_agehpnas_129 --workflow covertype_agehpnas_129 --app AgEHPNAS --args "--problem nas_big_data.covertype.problem_agebov4_skopt.Problem --max-evals 1000 --num-threads-per-rank 16 --num-ranks-per-node 8"
```

Finally, launch this experiment (for Theta):

```bash
balsam submit-launch -n 129 -t 180 -A $project_name -q default --job-mode mpi --wf-filter covertype_agehpnas_129
```

### Bayesian Optimization for joint optimization

Bayesian Optimization to optimize both hyperparameters and neural architectures, local testing with dummy evaluation function:

```bash
python -m nas_big_data.search.bo_hpo_nas --run nas_big_data.run.quick.run --problem nas_big_data.covertype.problem_agebov4_skopt.Problem --max-evals 1000
```

Create the `BO` application for Balsam on Theta:

```bash
balsam app --name BO --exe "$(which python) -m nas_big_data.search.bo_hpo_nas --evaluator balsam --run deephyper.nas.run.horovod.run"
```

Then create a job for a specific experiment:

```bash
balsam job --name covertype_bo_129 --workflow covertype_bo_129 --app BO --args "--problem nas_big_data.covertype.problem_agebov4_skopt.Problem --max-evals 1000 --num-threads-per-rank 16 --num-ranks-per-node 8"
```

Finally, launch this experiment (for Theta):

```bash
balsam submit-launch -n 129 -t 180 -A $project_name -q default --job-mode mpi --wf-filter covertype_bo_129
```
