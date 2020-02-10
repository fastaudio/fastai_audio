#!/bin/bash
# ------------------------------------------------------------------
# [FastAI SF Study Group] FastAI Audio Install Script
# ------------------------------------------------------------------

set -m
sudo apt-get --assume-yes install ffmpeg sox libsox-dev libsox-fmt-all
pip install -r requirements.txt