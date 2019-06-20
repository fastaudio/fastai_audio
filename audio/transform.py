from .audio import *
from .data import *

import matplotlib.pyplot as plt
import torch
from fastai import *
from fastai.text import *
from fastai.vision import *
import torch
import librosa
import torchaudio
from librosa.effects import split
from torchaudio import transforms
from scipy.signal import resample_poly


#Code altered from a kaggle kernel shared by @daisukelab, scales a spectrogram
#to be floats between 0 and 1 as this is how most 3 channel images are handled
def standardize(mel, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    mean = mean or mel.mean()
    std = std or mel.std()
    mel_std = (mel - mean) / (std + eps)
    _min, _max = mel_std.min(), mel_std.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    # Scale to [0, 1]
    if (_max - _min) > eps:
        V = mel_std
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = (V - norm_min) / (norm_max - norm_min)
    else: V = torch.zeros_like(mel_std)    
    return V

def torchdelta(mel, order=1, width=9):
    if(mel.shape[1] < width): 
        raise ValueError(f'''Delta not possible with current settings, inputs must be wider than 
        {width} columns, try setting max_to_pad to a larger value to ensure a minimum width''')
    return torch.from_numpy(librosa.feature.delta(mel.numpy(), order=order, width=9))

def tfm_sg_roll(spectro, max_shift_pct=0.7, direction=0, **kwargs):
    '''Shifts spectrogram along x-axis wrapping around to other side'''
    if len(spectro.shape) < 2:
        raise Exception('You are trying to apply the tranform to as signal')
    if int(direction) not in [-1, 0, 1]: 
        raise ValueError("Direction must be -1(left) 0(bidirectional) or 1(right)")
    direction = random.choice([-1, 1]) if direction == 0 else direction
    
    sg = spectro.clone()
    width = sg.shape[1]
    roll_by = int(width*random.random()*max_shift_pct*direction)
    sg = sg.roll(roll_by, dims=2)
    return sg

def tfm_sg_crop(spectro, crop_duration, sr, hop):
    if sr is None: return spectro
    sg = spectro.clone()
    c, y, x = sg.shape
    
    total_duration = (hop*x)/sr
    if crop_duration >= total_duration: return spectro
    crop_width = int(x*crop_duration/total_duration)
    
    crop_start = random.randint(0, x-crop_width)
    sg_crop = sg[:,:,crop_start:crop_start+crop_width]
    return sg_crop

def tfm_mask_time(spectro, tmasks=1, num_cols=20, start_col=None, tmask_value=None, **kwargs):
    '''Google SpecAugment time masking from https://arxiv.org/abs/1904.08779.'''
    sg = spectro.clone().squeeze(0)
    mask_value = sg.mean() if tmask_value is None else tmask_value
    x, y = sg.shape
    for _ in range(tmasks):
        mask = torch.ones(x, num_cols) * mask_value
        if start_col is None: start_col = random.randint(0, y-num_cols)
        if not 0 <= start_col <= y-num_cols: 
            raise ValueError(f"start_col value '{start_col}' out of range for sg of shape {sg.shape}")
        sg[:,start_col:start_col+num_cols] = mask
        start_col = None
    return sg.unsqueeze(0)

def tfm_mask_frequency(spectro, fmasks=1, num_rows=30, start_row=None, fmask_value=None, **kwargs):
    '''Google SpecAugment frequency masking from https://arxiv.org/abs/1904.08779.'''
    sg = spectro.clone().squeeze(0)
    mask_value = sg.mean() if fmask_value is None else fmask_value
    x, y = sg.shape
    for _ in range(fmasks):
        mask = torch.ones(num_rows, y) * mask_value
        if start_row is None: start_row = random.randint(0, x-num_rows)
        if not 0 <= start_row <= x-num_rows: 
            raise ValueError(f"start_row value '{start_row}' out of range for sg of shape {sg.shape}")
        sg[start_row:start_row+num_rows,:] = mask
        start_hori = None
    return sg.unsqueeze(0)

def get_spectro_transforms(mask_time:bool=True,
                           mask_frequency:bool=True,
                           roll:bool=True,
                           xtra_tfms:Optional[Collection[Transform]]=None,
                         **kwargs)->Collection[Transform]:
    "Utility func to create a list of spectrogram transforms"
    res = []
    if mask_time: res.append(partial(tfm_mask_time, **kwargs))
    if mask_frequency: res.append(partial(tfm_mask_frequency, **kwargs))
    if roll: res.append(partial(tfm_sg_roll, **kwargs))
    
    return (res+listify(xtra_tfms), [])

def tfm_trim_silence(signal, rate, threshold=20, pad_ms=200):
    '''Remove silence from start and end of audio'''
    actual = signal.clone().squeeze()
    padding = int(pad_ms/1000*rate)
    splits = split(actual.numpy(), top_db=threshold)
    return actual[splits[0, 0]-padding:splits[-1, -1]+padding].unsqueeze(0)

def tfm_chop_silence(signal, rate, threshold=20, pad_ms=200):
    '''Split signal at points of silence greater than 2*pad_ms '''
    actual = signal.clone().squeeze()
    padding = int(pad_ms/1000*rate)
    if(padding > len(actual)): return [actual]
    splits = split(actual.numpy(), top_db=threshold, hop_length=padding)
    return [actual[(max(a-padding,0)):(min(b+padding,len(actual)))] for (a, b) in splits]

def tfm_resample(signal, sr, sr_new):
    '''Resample using faster polyphase technique and avoiding FFT computation'''
    if(sr == sr_new): return signal
    sig_np = signal.squeeze(0).numpy()
    sr_gcd = math.gcd(sr, sr_new)
    resampled = resample_poly(sig_np, int(sr_new/sr_gcd), int(sr/sr_gcd))
    return torch.from_numpy(resampled).unsqueeze(0)

def tfm_shift(ai:AudioItem, max_pct=0.2):
    v = (.5 - random.random())*max_pct*len(ai.sig)
    sig = shift(ai.sig, v, cval=.0)
    sig = torch.tensor(sig)
    return AudioItem(sig=sig, sr=ai.sr)

def tfm_add_white_noise(ai:AudioItem, noise_scl=0.005, **kwargs)->AudioItem:
    noise = torch.randn_like(ai.sig) * noise_scl
    return AudioItem(ai.sig + noise, ai.sr)

def tfm_modulate_volume(ai:AudioItem, lower_gain=.1, upper_gain=1.2, **kwargs)->AudioItem:
    modulation = random.uniform(lower_gain, upper_gain)
    return AudioItem(ai.sig * modulation, ai.sr)

def tfm_random_cutout(ai:AudioItem, pct_to_cut=.15, **kwargs)->AudioItem:
    """Randomly replaces `pct_to_cut` of signal with silence. Similar to grainy radio."""
    copy = ai.sig.clone()
    mask = (torch.rand_like(copy)>pct_to_cut).float()
    masked = copy * mask
    return AudioItem(masked,ai.sr)

def tfm_pad_with_silence(ai:AudioItem, pct_to_pad=.15, min_to_pad=None, max_to_pad=None, **kwargs)->AudioItem:
    """Adds silence to beginning or end of signal, simulating microphone cut at start of end of audio."""
    if max_to_pad is None: max_to_pad = int(ai.sig.shape[0] * 0.15)
    if min_to_pad is None: min_to_pad = -max_to_pad
    pad = random.randint(min_to_pad, max_to_pad)
    copy = ai.sig.clone()
    if pad >= 0: copy[0:pad] = 0
    else: copy[pad:] = 0
    return AudioItem(copy,ai.sr)

def tfm_pitch_warp(ai:AudioItem, shift_by_pitch=None, bins_per_octave=12, **kwargs)->AudioItem:
    """CAUTION - slow!"""
    min_len = 600 # librosa requires a signal of length at least 500
    copy = ai.sig.clone()
    if (copy.shape[0] < min_len): copy = torch.cat((copy, torch.zeros(min_len - copy.shape[0])))
    if shift_by_pitch is None: shift_by_pitch = random.uniform(-3, 3)
    sig = torch.tensor(librosa.effects.pitch_shift(np.array(copy), ai.sr, shift_by_pitch, bins_per_octave))
    return AudioItem(sig,ai.sr)

def tfm_down_and_up(ai:AudioItem, sr_divisor=2, **kwargs)->AudioItem:
    """CAUTION - slow!"""
    copy = np.array(ai.sig.clone())
    down = librosa.audio.resample(copy, ai.sr, ai.sr/sr_divisor)
    sig = torch.tensor(librosa.audio.resample(down, ai.sr/sr_divisor, ai.sr))
    return AudioItem(sig,ai.sr)

def tfm_pad_to_max(ai:AudioItem, mx=1000):
    """Pad tensor with zeros (silence) until it reaches length `mx`"""
    copy = ai.sig.clone()
    padded = torchaudio.transforms.PadTrim(max_len=mx)(copy[None,:]).squeeze()
    return AudioItem(padded, ai.sr)

def tfm_pad_or_trim(ai:AudioItem, mx, trim_section="mid", pad_at_end=True, **kwargs):
    """Pad tensor with zeros (silence) until it reaches length `mx` frames, or trim clip to length `mx` frames"""
    sig = ai.sig.clone()
    siglen = len(sig)
    if siglen < mx:
        diff = mx - siglen
        padding = sig.new_zeros(diff) # Maintain input tensor device & type params
        nsig = torch.cat((sig,padding)) if pad_at_end else torch.cat((padding,sig))
    else:
        if trim_section not in {"start","mid","end"}:
            raise ValueError(f"'trim_section' argument must be one of 'start', 'mid' or 'end', got '{trim_section}'")
        if trim_section == "mid":
            nsig = sig.narrow(0, (siglen // 2) - (mx // 2), mx)
        elif trim_section == "end":
            nsig = sig.narrow(0, siglen-mx, mx)
        else:
            nsig = sig.narrow(0, 0, mx)
    return AudioItem(sig=nsig, sr=ai.sr)

def get_signal_transforms(white_noise:bool=True,
                         shift_max_pct:float=.6,
                         modulate_volume:bool=True,
                         random_cutout:bool=True,
                         pad_with_silence:bool=True,
                         pitch_warp:bool=True,
                         down_and_up:bool=True,
                         mx_to_pad:int=1000,
                         xtra_tfms:Optional[Collection[Transform]]=None,
                         **kwargs)->Collection[Transform]:
    "Utility func to easily create a list of audio transforms."
    res = []
    if shift_max_pct: res.append(partial(tfm_shift, max_pct=shift_max_pct))
    if white_noise: res.append(partial(tfm_add_white_noise, **kwargs))
    if modulate_volume: res.append(partial(tfm_modulate_volume, **kwargs))
    if random_cutout: res.append(partial(tfm_random_cutout, **kwargs))
    if pad_with_silence: res.append(partial(tfm_pad_with_silence, **kwargs))
    if pitch_warp: res.append(partial(tfm_pitch_warp, **kwargs))
    if down_and_up: res.append(partial(tfm_down_and_up, **kwargs))
    res.append(partial(tfm_pad_to_max, mx=mx_to_pad))
    if spectro: final_transform = partial(tfm_spectro, **kwargs)
    res.append(final_transform)
    #       train                   , valid
    return (res + listify(xtra_tfms), [partial(tfm_pad_to_max, mx=mx_to_pad)])