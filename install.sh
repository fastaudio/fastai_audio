#!/bin/bash
# ------------------------------------------------------------------
# [FastAI SF Study Group] FastAI Audio Install Script
# ------------------------------------------------------------------

set -m
apt-get update -y
apt-get --assume-yes install ffmpeg sox libsox-dev libsox-fmt-all libsndfile1
pip install -r requirements.txt

