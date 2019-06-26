import pytest
from fastai.gen_doc.doctest import this_tests
from os.path import abspath, dirname, join
import sys
sys.path.insert(1, abspath(dirname(dirname(__file__))))
from audio import *

@pytest.fixture(scope="module")
def path():
    data_url = 'http://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS'
    data_folder = datapath4file(url2name(data_url))
    if not os.path.exists(data_folder): untar_data(data_url, dest=data_folder)
    return data_folder

@pytest.fixture(scope="module")
def audio_list(path):
    return AudioList.from_folder(path)

@pytest.fixture(scope="module")
def random_item(path):
    return AudioItem(path=random.choice(path.ls()))

def test_path_can_be_str_type(path):
    #this_tests(AudioList.from_folder)
    assert AudioList.from_folder(str(path))

def test_cache_resample(random_item):
    #this_tests(resample_item)
    rs = 8000
    p = random_item.path
    config = AudioConfig(cache=True, resample_to=rs)
    item = (p, "Label Not Important")
    path_resample = config.cache_dir / f"sh_{md5(str(p)+str(rs))}"
    if os.path.exists(path_resample): os.remove(path_resample)
    assert not os.path.exists(path_resample)
    files = resample_item(item, config, p)
    for f, _ in files:
        assert os.path.exists(f)
        assert os.path.isfile(f)
        assert torchaudio.load(f)

def test_cache_silence(random_item):
    #this_tests(remove_silence)
    st, sp = 20, 200
    p = random_item.path
    config = AudioConfig(cache=True, silence_threshold=st, silence_padding=sp)
    item = (p, "Label Not Important")
    path_silence = config.cache_dir / f"sh_{md5(str(p)+str(st)+str(sp))}"
    files = remove_silence(item, config, p)
    for f, _ in files:
        assert os.path.exists(f)
        assert os.path.isfile(f)
        assert torchaudio.load(f)

def test_cache_segment(random_item):
    #this_tests(segment_items)
    segsize = 500
    p = random_item.path
    config = AudioConfig(cache=True, segment_size=segsize)
    label = "Label Not Important"
    item = (p, label)
    path_segment = config.cache_dir / f"s_{md5(str(p)+str(segsize)+str(label))}"
    files = segment_items(item, config, p)
    for f, _ in files:
        assert os.path.exists(f)
        assert os.path.isfile(f)
        assert torchaudio.load(f)

def test_can_open():
    p = Path('data/Right_whale.wav')
    AudioItem.open(p)


def test_can_from_folder():
    al = AudioList.from_folder('data')
    assert len(al) == 1

def test_can_from_df():
        df = pd.DataFrame(['Right_whale.wav']*2)
        al = AudioList.from_df(df, 'data')
        assert len(al) == 2
