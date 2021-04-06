#!/bin/bash -x
#COBALT -A datascience
#COBALT -n 1
#COBALT -q full-node
#COBALT -t 60

ACTIVATE_PYTHON_ENV="/lus/theta-fs0/projects/datascience/regele/thetagpu/testdh/SetUpEnv.sh"
echo "Script to activate Python env: $ACTIVATE_PYTHON_ENV"
source $ACTIVATE_PYTHON_ENV

python train.py