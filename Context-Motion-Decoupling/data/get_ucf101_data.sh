#!/usr/bin/env bash
# https://github.com/chaoyuaw/pytorch-coviar

DATA_DIR="./ucf101/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}
wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

unrar x UCF101.rar

wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip

unzip UCF101TrainTestSplits-RecognitionTask.zip
