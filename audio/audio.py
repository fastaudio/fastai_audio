from IPython.display import Audio
import mimetypes
import torchaudio
from torchaudio.transforms import PadTrim, DownmixMono
from fastai.data_block import ItemBase
from fastai.vision import Image
import numpy as np
import torch
import warnings
from pathlib import Path, PosixPath

AUDIO_EXTENSIONS = tuple(str.lower(k) for k, v in mimetypes.types_map.items()
                         if v.startswith('audio/'))

class AudioItem(ItemBase):
    def __init__(self, sig=None, sr=None, path=None, spectro=None, max_to_pad=None, start=None, end=None):
        '''Holds Audio signal and/or specrogram data'''
        if isinstance(sig, np.ndarray): sig = torch.from_numpy(sig)
        if sig is not None:
            if(len(sig.shape) == 1): sig = sig.unsqueeze(0)
            if(sig is not None and len(sig.shape) > 1 and sig.shape[0] > 1):
                warnings.warn(f'''Audio file {path} has {sig.shape[0]} channels, automatically downmixing to mono''')
                sig = DownmixMono(channels_first=True)(sig)
        self._sig, self._sr, self.path, self.spectro = sig, sr, path, spectro
        self.max_to_pad = max_to_pad
        self.start, self.end = start, end

    def __str__(self):
        return f'{self.__class__.__name__} {round(self.duration, 2)} seconds ({self.sig.shape[0]} samples @ {self.sr}hz)'

    def __len__(self): return self.data.shape[0]
    
    def _repr_html_(self):
        return f'{self.__str__()}<br />{self.ipy_audio._repr_html_()}'

    @classmethod
    def open(self, item, **kwargs):
        if isinstance(item, (Path, PosixPath, str)):
            sig, sr = torchaudio.load(item)
            return AudioItem(sig, sr, path=Path(item))
        if isinstance(item, (tuple, np.ndarray)):
            return AudioItem(item)

    def show(self, title: [str] = None, **kwargs):
        print(f"File: {self.path}")
        print(f"Total Length: {round(self.duration, 2)} seconds")
        self.hear(title=title)
        for im in self.get_spec_images(): display(im)                 
                         
    def get_spec_images(self):
        sg = self.spectro
        if sg is None: return [] 
        if torch.all(torch.eq(sg[0], sg[1])) and torch.all(torch.eq(sg[0], sg[2])):
            return [Image(sg[0].unsqueeze(0))]
        else: 
            return [Image(s.unsqueeze(0)) for s in sg]

    def hear(self, title=None):
        if title is not None: print("Label:", title)
        if self.start is not None or self.end is not None:
            print(f"{round(self.start/self.sr, 2)}s-{round(self.end/self.sr,2)}s of original clip")
            start = 0 if self.start is None else self.start
            end = len(self.sig)-1 if self.end is None else self.end
            display(Audio(data=self.sig[start:end], rate=self.sr))
        else:
            display(self.ipy_audio)
        

    def apply_tfms(self, tfms):
        for tfm in tfms:
            self.data = tfm(self.data)
        return self

    @property
    def shape(self): return self.data.shape

    def _reload_signal(self):
        sig, sr = torchaudio.load(self.path)
        if self.max_to_pad is not None:
            sig = tfm_padtrim_signal(sig, int(self.max_to_pad/1000*sr), pad_type="zeros")
        self._sr = sr
        self._sig = sig

    @property
    def sig(self):
        if not hasattr(self, '_sig') or self._sig is None:
            self._reload_signal()
        return self._sig.squeeze(0)
    
    @sig.setter
    def sig(self, sig): self._sig = sig

    @property
    def sr(self):
        if not hasattr(self, '_sr') or self._sr is None:
            si, ei = torchaudio.info(str(self.path))
            self._sr = si.rate
        return self._sr
    
    @sr.setter
    def sr(self, sr): self._sr = sr

    @property
    def ipy_audio(self): return Audio(data=self.sig, rate=self.sr)

    @property
    def duration(self): 
        if(self._sig is not None): return len(self.sig)/self.sr
        else: 
            si, ei = torchaudio.info(str(self.path))
            return si.length/si.rate
        
    @property
    def data(self): return self.spectro if self.spectro is not None else self.sig
    @data.setter
    def data(self, x):
        if self.spectro is not None: self.spectro = x
        else:                        self.sig = x
