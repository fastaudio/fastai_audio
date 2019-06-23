#!/bin/bash
# ------------------------------------------------------------------
# [FastAI SF Study Group] FastAI Audio Install Script
# ------------------------------------------------------------------

set -m
pip install pydub librosa fire --user
sudo apt-get --assume-yes install ffmpeg sox libsox-dev libsox-fmt-all
pip install git+https://github.com/pytorch/audio.git@d92de5b97fc6204db4b1e3ed20c03ac06f5d53f0
