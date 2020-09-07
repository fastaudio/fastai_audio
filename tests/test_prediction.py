import torch

import pytest
import torchaudio
from fastai.basic_data import Path

from audio import AudioConfig, AudioDataBunch, AudioList, audio_learner
from audio.learner import audio_predict
from audio.util import create_sin_wave

import pytest


@pytest.fixture(scope='session')
def audio_db():
    sig, sr = create_sin_wave()
    folder = './data/misc/test'
    for i in range(5):
        torchaudio.save(f"{folder}/test{i}.wav", sig, sr)
    config = AudioConfig()
    audios = AudioList.from_folder(
        folder, config=config).split_by_rand_pct(.2, seed=4).label_from_func(lambda x: '1')
    tfms = None
    db = audios.transform(tfms).databunch(bs=2)
    return db


def pred_correct(pred):
    assert(len(pred) == 3)
    assert isinstance(pred[1], torch.Tensor)
    assert isinstance(pred[2], torch.Tensor)


def test_predict_path(audio_db):
    folder = './data/misc/test'
    learn = audio_learner(audio_db)
    pred = audio_predict(learn, f"{folder}/test1.wav")
    pred_correct(pred)


def test_predict_audio_item(audio_db):
    learn = audio_learner(audio_db)
    pred = audio_predict(learn, audio_db.train_dl.dataset[0][0])
    pred_correct(pred)
