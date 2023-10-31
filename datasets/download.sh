#!/bin/bash

if [ -z $1 ];then
  echo "Download datasets..."
  gdown --folder --id 1ELzsdlc1Bd5C97EHn5P_k8qqM7ZKEFHF
  cd datasets
  unzip blender.zip && rm blender.zip
  unzip dmsr.zip && rm dmsr.zip
  unzip scannet.zip && rm scannet.zip
elif [ $1 = blender ];then
  mkdir -p datasets && cd datasets
  echo "Download dataset: Blender..."
  gdown 1xjGKFszIP8dX7i_kOFq3RFZ7tSJHzPQM
  unzip blender.zip && rm blender.zip
elif [ $1 = dmsr ];then
  mkdir -p datasets && cd datasets
  echo "Download dataset: DM-SR..."
  gdown 14bxsM1a9QnP9b7GHBFuU6ln1nq03Qsy9
  unzip dmsr.zip && rm dmsr.zip
elif [ $1 = scannet ];then
  mkdir -p datasets && cd datasets
  echo "Download dataset: ScanNet..."
  gdown 1UzJzcgBkGo6KfhZMLFCikXoaXptbWPF-
  unzip scannet.zip && rm scannet.zip
fi