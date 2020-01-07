#!/bin/bash
# ------------------------------------------------------------------
# [FastAI SF Study Group] FastAI Audio Install Script
# ------------------------------------------------------------------

set -m
pip install pydub librosa fire --user
sudo apt-get --assume-yes install ffmpeg sox libsox-dev libsox-fmt-all
pip install torchaudio==0.3.0