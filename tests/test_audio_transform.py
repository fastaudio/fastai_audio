import pytest
from fastai.gen_doc.doctest import this_tests
from fastai.vision import *
import sys, os
sys.path.append('..')
from audio import *

@pytest.fixture(scope="module")
def path():
    data_url = 'http://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS'
    data_folder = datapath4file(url2name(data_url))
    if not os.path.exists(data_folder): untar_data(data_url, dest=data_folder)
    return data_folder

@pytest.fixture(scope="module")
def random_item(path):
    return open_audio(random.choice(path.ls()))

@pytest.fixture(scope="module")
def fixed_item(path):
    return open_audio(path.ls()[0])

def test_resample(random_item):
    #this_tests(tfm_resample)
    sig, sr = random_item.sig, random_item.sr
    input_samples = len(sig.squeeze(0))
    new_srs = [44100, 22050, 16000, 8000]
    for new_sr in new_srs:
        expected_output = round(input_samples*new_sr/sr)
        resampled = tfm_resample(sig, sr, new_sr)
        assert len(resampled.squeeze(0)) == expected_output, f"Failed for sr: {new_sr}"
    