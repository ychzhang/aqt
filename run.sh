#!/bin/bash
VM_NAME=v3vm
ZONE=us-east1-d
LOCAL_WORK_DIR=/home/yichi/work_dir
CONFIG_DICT=configs/pokebnn/pokebnn_config.py
TMUX_SESSION=poke
COMMAND="
export TF_CPP_MIN_LOG_LEVEL=0 &&
export PYTHONPATH=/home/yichi/aqt &&
export TFDS_DATA_DIR=gs://bin-research-data/dataset/imagenet2012 &&
cd aqt/aqt/jax_legacy/jax/imagenet &&
python3 train.py --model_dir $LOCAL_WORK_DIR --hparams_config_dict $CONFIG_DICT --batch_size 8192 --resnet508b_ckpt_path gs://bin-research-data/logs/aqt-resnet50-w8a8auto-64v3 --config_idx 0 2>&1 | tee -a /home/yichi/training_log.txt
"

gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all --command "tmux new-session -d -s $TMUX_SESSION '$COMMAND'"
#gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all --command "$COMMAND"

