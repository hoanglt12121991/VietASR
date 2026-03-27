#!/bin/bash

pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install k2==1.24.4.dev20250715+cuda12.1.torch2.5.1 -f https://k2-fsa.github.io/k2/cuda.html
pip install git+https://github.com/lhotse-speech/lhotse
pip install kaldifeat==1.25.5.dev20250203+cuda12.1.torch2.5.1 -f https://csukuangfj.github.io/kaldifeat/cuda.html

git clone https://github.com/k2-fsa/icefall ./icefall
cd ./icefall
git pull
pip install -r requirements.txt

python download_checkpoint.py ./models/viet_iter3_pseudo_label