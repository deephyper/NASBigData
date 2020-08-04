#!/bin/bash

cd /projects/datascience/regele/
conda activate dh-env/
cd /projects/datascience/regele/NASBigData/nas_big_data/cifar10/best/agebov3_8_test_augment_f32_n4_r1_5/
python -m nas_big_data.cifar10.best.agebov3_8_test_augment_f32_n4_r1_5
