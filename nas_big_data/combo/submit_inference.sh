#!/bin/bash -x
#COBALT -A datascience
#COBALT -n 1
#COBALT -q full-node
#COBALT -t 30

ACTIVATE_PYTHON_ENV="/lus/grand/projects/datascience/regele/thetagpu/agebo/SetUpEnv.sh"
echo "Script to activate Python env: $ACTIVATE_PYTHON_ENV"
source $ACTIVATE_PYTHON_ENV

python inference_times.py