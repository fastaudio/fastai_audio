from pathlib import PosixPath
from IPython.core.debugger import set_trace
import os
from dataclasses import dataclass, asdict
import hashlib

from fastai.vision import *
import torchaudio
from torchaudio.transforms import MelSpectrogram, SpectrogramToDB
from .audio import *
from .transform import *

def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def preview_transforms(items, path, config):
    if not isinstance(items, (list, np.ndarray)):
        items = [items]
    audios = AudioList(items, path, config=config).split_none().label_empty()
    return audios.train    

class AudioDataBunch(DataBunch):
    def show_batch(self, rows: int = 3, ds_type: DatasetType = DatasetType.Train, **kwargs):
        batch = self.dl(ds_type).dataset[:rows]
        prev = None
        for x, y in batch:
            print(y)
            x.show()

@dataclass
class SpectrogramConfig:
    '''Configuration for how Spectrograms are generated'''
    n_fft: int = 1024
    ws: int = None
    hop: int = 72
    f_min: int = 0
    f_max: int = 8000
    pad: int = 0
    n_mels: int = 224

@dataclass
class AudioTransformConfig:
    '''Options for pre-processing audio signals'''
    remove_silence: bool = False
    use_spectro: bool = True
    cache: bool = True
    cache_dir = Path('.cache')
    force_cache = False
    to_db_scale = True
    silence_padding: int = 200
    top_db: int = 80
    processed = False
    segment_size: int = None
    window_size: int = None
    silence_threshold: int = 20
    max_to_pad: float = None
    sg_cfg = SpectrogramConfig()
    resample_to: int = None

def get_cache(config, cache_type, hash_params):
    if not config.cache_dir: return None
    hash_str = md5("".join(map(str, hash_params)))
    mark = config.cache_dir / f"{cache_type}_{hash_str}"
    files = get_files(mark) if mark.exists() else None
    return files

def make_cache(sigs, sr, config, cache_type, hash_params):
    hash_str = md5("".join(map(str, hash_params)))
    mark = config.cache_dir / f"{cache_type}_{hash_str}"
    files = []
    if len(sigs) > 0:
        os.makedirs(mark, exist_ok=True)
        for i, s in enumerate(sigs):
            if len(s) < 1: continue
            fn = mark/(str(i) + '.wav')
            files.append(fn)
            torchaudio.save(str(fn), s, sr)
    return files

def resample_item(item, config):
    item_path, label = item
    sr_new = config.resample_to
    # print("asdsad", item_path)
    files = get_cache(config, "rs", [item_path, sr_new])
    if not files:
        ai = AudioItem.open(item_path)
        sig, sr = ai.sig, ai.sr
        sig = [tfm_resample(sig, sr, sr_new)]
        files = make_cache(sig, sr_new, config, "rs", [item_path, sr_new])
    return list(zip(files, [label]*len(files)))

def remove_silence(item, config):
    item_path, label = item
    st, sp = config.silence_threshold, config.silence_padding
    files = get_cache(config, "sh", [item_path, st, sp])
    if not files:
        ai = AudioItem.open(item_path)
        sig, sr = ai.sig, ai.sr
        sigs = tfm_chop_silence(sig, sr, st, sp)
        files = make_cache(sigs, sr, config, "sh", [item_path, st, sp])
    return list(zip(files, [label]*len(files)))

def segment_items(item, config):
    item_path, label = item
    ai = AudioItem.open(item_path)
    sig, sr = ai.sig, ai.sr
    segsize = int(config.segment_size / 1000 * sr)
    window_size = segsize if config.window_size is None else int(config.window_size / 1000 * sr)
    files = get_cache(config, "s", [item_path, segsize, label])
    if not files:
        sig = sig.squeeze()
        sigs = []
        i = 0
        if len(sig) < segsize:
            sigs.append(sig)
        else:
            while (i + segsize) < len(sig):
                sigs.append(sig[i:i+segsize])
                i += config.window_size
        files = make_cache(sigs, sr, config, "s", [item_path, segsize, label])
    return list(zip(files, [label]*len(files)))

class AudioLabelList(LabelList):

    def _pre_process(self):
        x, y = self.x, self.y
        cfg = x.config
        if len(x.items) > 0 and (cfg.remove_silence or cfg.segment_size or cfg.resample_to):
            items = []
            for i in x.items:
                if isinstance(i, (PosixPath, Path, str)):
                    if str(x.path) not in str(i):
                        items.append(x.path/i)
                        continue
                elif str(x.path) not in str(i.path):
                    i.path = x.path/i.path                
                items.append(i)
            x.inner_df = None
            items = list(zip(items, y.items))
            def concat(x, y): return np.concatenate(
                (x, y)) if len(y) > 0 else x
            
            if x.config.resample_to:
                items = [resample_item(i, cfg) for i in items]
                items = reduce(concat, items, np.empty((0, 2)))
            
            
            if x.config.remove_silence:
                items = [remove_silence(i, cfg) for i in items]
                items = reduce(concat, items, np.empty((0, 2)))
            
            if x.config.segment_size:
                items = [segment_items(i, cfg) for i in items]
                items = reduce(concat, items, np.empty((0, 2)))


            nx, ny = tuple(zip(*items))
            x.items, y.items = np.array(nx), np.array(ny)
            
        self.x, self.y = x, y
        self.y.x = x

    def process(self, *args, **kwargs):
        self._pre_process()
        super().process(*args, **kwargs)


class AudioList(ItemList):
    _bunch = AudioDataBunch
    config: AudioTransformConfig

    def __init__(self, items, path, config=AudioTransformConfig(), **kwargs):
        super().__init__(items, path, **kwargs)
        self._label_list = AudioLabelList
        config.cache_dir = path / config.cache_dir
        self.config = config
        self.copy_new += ['config']

    def open(self, item) -> AudioItem:
        p = Path(item)
        if not p.exists():                                raise Exception('File not found: ' + p)
        if not str(p).lower().endswith(AUDIO_EXTENSIONS): raise Exception("Invalid audio file")

        cfg = self.config
        if cfg.use_spectro:
            cache_dir = self.path / cfg.cache_dir
            s = md5(str(asdict(cfg)) + str(asdict(cfg.sg_cfg)) + str(p))
            image_path = cache_dir/(f"{s}.pt")
            if cfg.cache and not cfg.force_cache and image_path.exists():
                spectro = torch.load(image_path)
                return AudioItem(spectro=spectro, path=item, max_to_pad=cfg.max_to_pad)

        signal, samplerate = torchaudio.load(str(p))

        if cfg.max_to_pad:
            signal = PadTrim(max_len=int(cfg.max_to_pad/1000*samplerate))(signal)

        mel = None
        if cfg.use_spectro:
            mel = MelSpectrogram(**asdict(cfg.sg_cfg))(signal.reshape(1, -1))
            mel = mel.permute(0, 2, 1)
            if cfg.to_db_scale:
                mel = SpectrogramToDB(top_db=cfg.top_db)(mel)
            if cfg.cache:
                os.makedirs(image_path.parent, exist_ok=True)
                torch.save(mel, image_path)
        return AudioItem(sig=signal.squeeze(), sr=samplerate, spectro=mel, path=item)

    def get(self, i):
        item = self.items[i]
        
        if isinstance(item, (PosixPath, Path, str)):
            return self.open(self.path/item)

        if isinstance(item, AudioItem):
            return item

        raise Exception("Can't handle that type")

    @classmethod
    def from_folder(cls, path: PathOrStr = '.', extensions: Collection[str] = None, recurse: bool = True, **kwargs) -> ItemList:
        if not extensions:
            extensions = AUDIO_EXTENSIONS
        return cls(get_files(path, extensions, recurse), path, **kwargs)
