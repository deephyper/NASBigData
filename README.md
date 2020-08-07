# NASBigData
Neural architecture search for big data problems

## Installation

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
