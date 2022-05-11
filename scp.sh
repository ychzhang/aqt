#!/bin/bash
VM_NAME=v3vm
ZONE=us-east1-d

mkdir /home/yichi/test
cp -r /home/yichi/aqt /home/yichi/test/aqt
rm -rf /home/yichi/test/aqt/.git
gcloud alpha compute tpus tpu-vm scp /home/yichi/test/aqt $VM_NAME: --worker=all --zone=$ZONE --recurse
rm -r /home/yichi/test

