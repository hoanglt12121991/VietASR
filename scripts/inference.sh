#!/bin/bash

export PATH=/home/ubuntu/HoangLT19/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/home/ubuntu/HoangLT19/cuda-12.1/lib64:$LD_LIBRARY_PATH

export PYTHONPATH=$(pwd)/icefall:$PYTHONPATH

python ./ASR/zipformer/pretrained.py \
  --checkpoint ./models/viet_iter3_pseudo_label/exp/epoch-12.pt \
  --tokens ./models/viet_iter3_pseudo_label/data/Vietnam_bpe_2000_new/tokens.txt \
  /home/ubuntu/HoangLT19/meeting-summarizer/backend/exp/dataset/mp3/FPTOpenSpeechData_Set001_V0.1_000010.mp3 \
  /home/ubuntu/HoangLT19/meeting-summarizer/backend/exp/dataset/mp3/FPTOpenSpeechData_Set001_V0.1_000011.mp3 \
  /home/ubuntu/HoangLT19/meeting-summarizer/backend/exp/dataset/mp3/FPTOpenSpeechData_Set001_V0.1_000012.mp3 \
  /home/ubuntu/HoangLT19/meeting-summarizer/backend/exp/dataset/mp3/FPTOpenSpeechData_Set001_V0.1_000013.mp3 \
  /home/ubuntu/HoangLT19/meeting-summarizer/backend/exp/dataset/mp3/FPTOpenSpeechData_Set001_V0.1_000014.mp3 \
  /home/ubuntu/HoangLT19/meeting-summarizer/backend/exp/dataset/mp3/FPTOpenSpeechData_Set001_V0.1_000015.mp3 \
  --method modified_beam_search \

#   --method greedy_search \
#   --method fast_beam_search \
#   --causal 1 \
#   --chunk-size 16 \
#   --left-context-frames 128 \