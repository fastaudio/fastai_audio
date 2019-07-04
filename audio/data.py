from pathlib import Path as PosixPath
from IPython.core.debugger import set_trace
import os
from collections import Counter
from dataclasses import dataclass, asdict
import hashlib
import matplotlib as plt
from fastai.vision import *
from fastprogress import progress_bar
import torchaudio
import warnings
from torchaudio.transforms import MelSpectrogram, SpectrogramToDB, MFCC
from .audio import *
from .transform import *

def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()

class AudioDataBunch(DataBunch):
    def show_batch(self, rows: int = 3, ds_type: DatasetType = DatasetType.Train, **kwargs):
        batch = self.dl(ds_type).dataset[:rows]
        prev = None
        for x, y in batch:
            print('-'*60)
            x.show(title=y)

@dataclass
class SpectrogramConfig:
    '''Configuration for how Spectrograms are generated'''
    f_min: int = 0
    f_max: int = 22050
    hop: int = 256
    n_fft: int = 2560
    n_mels: int = 128
    pad: int = 0
    to_db_scale: bool = True
    top_db: int = 100
    ws: int = None
    n_mfcc: int = 20
    def mel_args(self):
        return {k:v for k, v in asdict(self).items() if k in ["f_min", "f_max", "hop", "n_fft", 
                                                      "n_mels", "pad", "ws"]}
        
@dataclass
class AudioConfig:
    '''Options for pre-processing audio signals'''
    cache: bool = True
    cache_dir = Path('.cache')
    force_cache = False
    
    duration: int = None
    max_to_pad: float = None
    remove_silence: str = None
    use_spectro: bool = True
    mfcc: bool = False
    
    delta: bool = False
    silence_padding: int = 200
    silence_threshold: int = 20
    segment_size: int = None
    resample_to: int = None
    standardize: bool = False
    _processed = False
    _sr = None
    sg_cfg: SpectrogramConfig = SpectrogramConfig()
        
    def __setattr__(self, name, value):
        if name in 'duration max_to_pad segment_size'.split():
            if value is not None and value <= 30:
                warnings.warn(f"{name} should be in milliseconds, it looks like you might be trying to use seconds")
        super(AudioConfig, self).__setattr__(name, value)
    
def get_cache(config, cache_type, item_path, params):
    if not config.cache_dir: return None
    details = "-".join(map(str, params))
    top_level = config.cache_dir / f"{cache_type}_{details}"
    subfolder = f"{item_path.name}-{md5(str(item_path))}"
    mark = top_level/subfolder
    files = get_files(mark) if mark.exists() else None
    return files

def make_cache(sigs, sr, config, cache_type, item_path, params):
    details = "-".join(map(str, params))
    top_level = config.cache_dir / f"{cache_type}_{details}"
    subfolder = f"{item_path.name}-{md5(str(item_path))}"
    mark = top_level/subfolder
    files = []
    if len(sigs) > 0:
        os.makedirs(mark, exist_ok=True)
        for i, s in enumerate(sigs):
            if len(s) < 1: continue
            fn = mark/(str(i) + '.wav')
            files.append(fn)
            torchaudio.save(str(fn), s, sr)
    return files

def resample_item(item, config, path):
    item_path, label = item
    if not os.path.exists(item_path): item_path = path/item_path
    sr_new = config.resample_to
    files = get_cache(config, "rs", item_path, [sr_new])
    if not files:
        sig, sr = torchaudio.load(item_path)
        sig = [tfm_resample(sig, sr, sr_new)]
        files = make_cache(sig, sr_new, config, "rs", item_path, [sr_new])
    return list(zip(files, [label]*len(files)))

def remove_silence(item, config, path):
    item_path, label = item
    if not os.path.exists(item_path): item_path = path/item_path
    st, sp = config.silence_threshold, config.silence_padding
    remove_type = config.remove_silence
    cache_prefix = f"sh-{remove_type[0]}"
    files = get_cache(config, cache_prefix, item_path, [st, sp])
    if not files:
        sig, sr = torchaudio.load(item_path)
        sigs = tfm_remove_silence(sig, sr, remove_type, st, sp)
        files = make_cache(sigs, sr, config, cache_prefix, item_path, [st, sp])
    return list(zip(files, [label]*len(files)))

def segment_items(item, config, path):
    item_path, label = item
    if not os.path.exists(item_path): item_path = path/item_path
    sig, sr = torchaudio.load(item_path)
    segsize = int(sr*config.segment_size/1000)
    files = get_cache(config, "s", item_path, [config.segment_size])
    if not files:
        sig = sig.squeeze()
        sigs = []
        siglen = len(sig)
        for i in range((siglen//segsize) + 1):
            #if there is a full segment, add it, if not take the remaining part and zero pad to correct length
            if((i+1)*segsize <= siglen): sigs.append(sig[i*segsize:(i+1)*segsize])
            else: sigs.append(torch.cat([sig[i*segsize:], torch.zeros(segsize-len(sig[i*segsize:]))]))
        files = make_cache(sigs, sr, config, "s", item_path, [config.segment_size])
    return list(zip(files, [label]*len(files)))

def get_outliers(len_dict, devs):
    np_lens = array(list(len_dict.values()))
    stdev = np_lens.std()
    lower_thresh = np_lens.mean() - stdev*devs
    upper_thresh = np_lens.mean() + stdev*devs
    outliers = [(k,v) for k,v in len_dict.items() if not (lower_thresh < v < upper_thresh)]
    return sorted(outliers, key=lambda tup: tup[1])

def _set_sr(item_path, config, path):
    if not os.path.exists(item_path): item_path = path/item_path
    sig, sr = torchaudio.load(item_path)
    config._sr = sr

class AudioLabelList(LabelList):

    def _pre_process(self):
        x, y = self.x, self.y
        cfg = x.config
        
        if len(x.items) > 0:
            if not cfg.resample_to: _set_sr(x.items[0], x.config, x.path)
            if cfg.remove_silence or cfg.segment_size or cfg.resample_to:
                items = list(zip(x.items, y.items))

                def concat(x, y): return np.concatenate(
                    (x, y)) if len(y) > 0 else x
                
                if x.config.resample_to:
                    cfg._sr = x.config.resample_to 
                    items = [resample_item(i, x.config, x.path) for i in items]
                    items = reduce(concat, items, np.empty((0, 2)))

                if x.config.remove_silence:
                    items = [remove_silence(i, x.config, x.path) for i in items]
                    items = reduce(concat, items, np.empty((0, 2)))

                if x.config.segment_size:
                    items = [segment_items(i, x.config, x.path) for i in items]
                    items = reduce(concat, items, np.empty((0, 2)))

                nx, ny = tuple(zip(*items))
                x.items, y.items = np.array(nx), np.array(ny)
 
        self.x, self.y = x, y
        self.y.x = x
   
    def process(self, *args, **kwargs):
        self._pre_process()
        super().process(*args, **kwargs)
        self.x.config._processed = True

class AudioList(ItemList):
    _bunch = AudioDataBunch
    config: AudioConfig

    def __init__(self, items, path, config=AudioConfig(), **kwargs):
        super().__init__(items, path, **kwargs)
        self._label_list = AudioLabelList
        config.cache_dir = path / config.cache_dir
        self.config = config
        self.copy_new += ['config']

    def open(self, item) -> AudioItem:
        p = Path(item)
        if self.path is not None and str(self.path) not in str(item): p = self.path/item
        if not p.exists(): 
            raise FileNotFoundError(f"Neither '{item}' nor '{p}' could be found")
        if not str(p).lower().endswith(AUDIO_EXTENSIONS): raise Exception("Invalid audio file")

        cfg = self.config
        if cfg.use_spectro:
            cache_dir = self.path / cfg.cache_dir
            folder = md5(str(asdict(cfg))+str(asdict(cfg.sg_cfg)))
            fname = f"{md5(str(p))}-{p.name}.pt"
            image_path = cache_dir/(f"{folder}/{fname}")
            if cfg.cache and not cfg.force_cache and image_path.exists():
                mel = torch.load(image_path).squeeze()
                start, end = None, None
                if cfg.duration and cfg._processed:
                    mel, start, end = tfm_crop_time(mel, cfg._sr, cfg.duration, cfg.sg_cfg.hop)
                return AudioItem(spectro=mel, path=item, max_to_pad=cfg.max_to_pad, start=start, end=end)

        signal, samplerate = torchaudio.load(str(p))
        if(cfg._sr is not None and samplerate != cfg._sr):
            raise ValueError(f'''Multiple sample rates detected. Sample rate {samplerate} of file {str(p)} 
                                does not match config sample rate {cfg._sr} 
                                this means your dataset has multiple different sample rates, 
                                please choose one and set resample_to to that value''')
        if(signal.shape[0] > 1):
            warnings.warn(f"Audio file {p} is multichannel, automatically downmixing to mono")
            signal = DownmixMono(channels_first=True)(signal)
        if cfg.max_to_pad or cfg.segment_size:
            pad_len = cfg.max_to_pad if cfg.max_to_pad is not None else cfg.segment_size
            signal = PadTrim(max_len=int(pad_len/1000*samplerate))(signal)

        mel = None
        if cfg.use_spectro:
            if cfg.mfcc: mel = MFCC(sr=samplerate, n_mfcc=cfg.sg_cfg.n_mfcc, melkwargs=cfg.sg_cfg.mel_args())(signal)
            else:
                mel = MelSpectrogram(**(cfg.sg_cfg.mel_args()))(signal)
                if cfg.sg_cfg.to_db_scale: mel = SpectrogramToDB(top_db=cfg.sg_cfg.top_db)(mel)
            mel = mel.squeeze().permute(1, 0).flip(0)
            if cfg.standardize: mel = standardize(mel)
            if cfg.delta: mel = torch.stack([mel, torchdelta(mel), torchdelta(mel, order=2)]) 
            else: mel = mel.expand(3,-1,-1)
            if cfg.cache:
                os.makedirs(image_path.parent, exist_ok=True)
                torch.save(mel, image_path)
            start, end = None, None
            if cfg.duration and cfg._processed: 
                mel, start, end = tfm_crop_time(mel, cfg._sr, cfg.duration, cfg.sg_cfg.hop)
        return AudioItem(sig=signal.squeeze(), sr=samplerate, spectro=mel, path=item, start=start, end=end)

    def get(self, i):
        item = self.items[i]
        if isinstance(item, AudioItem):
            return item
        if isinstance(item, (PosixPath, Path)):
            return self.open(item)
        if isinstance(item, str):
            if not ('/') in item: return self.open(self.path/item)
            else:                 return self.open(item)

        raise Exception("Can't handle that type")     
        
    def stats(self, prec=0, devs=3, figsize=(15,5)):
        '''Displays samples, plots file lengths and returns outliers of the AudioList'''
        len_dict = {}
        rate_dict = {}
        pb = progress_bar(self)
        for item in pb:
            si, ei = torchaudio.info(str(item.path))
            len_dict[item.path] = si.length/si.rate
            rate_dict[item.path] = si.rate
        lens = list(len_dict.values())
        rates = list(rate_dict.values())
        print("Sample Rates: ")
        for sr,count in Counter(rates).items(): print(f"{int(sr)}: {count} files")
        self._plot_lengths(lens, prec, figsize)
        return len_dict 
    
    def _plot_lengths(self, lens, prec, figsize):
        '''Plots a list of file lengths displaying prec digits of precision'''
        rounded = [round(i, prec) for i in lens]
        rounded_count = Counter(rounded)
        plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
        labels = sorted(rounded_count.keys())
        values = [rounded_count[i] for i in labels]
        width = 1
        plt.bar(labels, values, width)
        xticks = np.linspace(int(min(rounded)), int(max(rounded))+1, 10)
        plt.xticks(xticks)
        plt.show()
   
    

    @classmethod
    def from_folder(cls, path: PathOrStr = '.', extensions: Collection[str] = None, recurse: bool = True, **kwargs) -> ItemList:
        if not extensions:
            extensions = AUDIO_EXTENSIONS
        return cls(get_files(path, extensions, recurse), path, **kwargs)
