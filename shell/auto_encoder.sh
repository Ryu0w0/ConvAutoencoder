#! /bin/bash
#SBATCH "--job-name=auto_en"
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output files/output/server_logs/job%J.out
#SBATCH --error files/output/server_logs/job%J.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

module load $1

# PARAMETERS
save_key="-save_key $3"
log_level="-log_level INFO"
use_gpu="-use_gpu 1"
is_reproducible="-is_reproducible 0"
is_local="-is_local 0"
# MODEL
model_config_key="-model_config_key $2"
# TRAINING
use_aug="-use_aug 1"
num_folds="-num_folds 5"
num_epoch="-num_epoch 200"
batch_size="-batch_size 64"
num_workers="-num_workers 1"
save_img_per_epoch="-save_img_per_epoch 5"
# TEST
do_test="-do_test 0"

python3 ./autoencoder_main.py $save_key $log_level $use_gpu $is_reproducible $is_local $model_config_key $use_aug $num_folds $num_epoch $batch_size $num_workers $save_img_per_epoch $do_test
