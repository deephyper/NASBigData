#!/bin/bash
#COBALT -t 20
#COBALT -n 3
#COBALT -A datascience
#COBALT -q debug-cache-quad

# CONFIGURABLE VARIABLES
EXP_FOLDER=covertype
WALLTIME=1200

# LOAD ENVIRONMENT
source ~/.bashrc
cd /projects/datascience/regele/
conda activate autogluon-env/

# MOVE TO EXP FOLDER
cd /projects/datascience/regele/NASBigData/nas_big_data/$EXP_FOLDER/autogluon/

# RUN EXPERIMENT
python -m nas_big_data.covertype.autogluon.first_node  | xargs ssh -T; python -m nas_big_data.covertype.autogluon.evaluate_theta_hpo --walltime $WALLTIME --no-knn