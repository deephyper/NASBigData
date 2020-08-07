#!/bin/bash
#COBALT -t 300
#COBALT -n 1
#COBALT -A datascience
#COBALT -q default

# CONFIGURABLE VARIABLES
EXP_FOLDER=dionis
WALLTIME=14400

# LOAD ENVIRONMENT
source ~/.bashrc
cd /projects/datascience/regele/
conda activate dh-env/

# MOVE TO EXP FOLDER
cd /projects/datascience/regele/NASBigData/nas_big_data/$EXP_FOLDER/autogluon/

# RUN EXPERIMENT
python -m nas_big_data.$EXP_FOLDER.autogluon.evaluate --walltime $WALLTIME
