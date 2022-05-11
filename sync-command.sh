#!/bin/bash
VM_NAME=v3vm
ZONE=us-east1-d
LOCAL_WORK_DIR=/home/yichi/work_dir/
GCS_WORK_DIR=gs://bin-research-data/logs/easy-poke/sweep-bop
COMMAND="
cp -r log work_dir/ && gsutil rsync -r -d $LOCAL_WORK_DIR $GCS_WORK_DIR
"

# ssh and execute command on VM
gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --worker=all --command "$COMMAND"
