# Fast AI Audio

This is an audio module built on top of FastAI to allow you to quickly and easily build machine learning models for a wide variety of audio applications. We are an unofficial library and have no official connection to fastai except that we are huge fans and want to help make their tools more widely available.

# Installation
To install, run the following commands in a shell. 

```
git clone https://github.com/mogwai/fastai_audio
cd fastai_audio
sudo chmod +x install.sh
./install.sh
```

If you wish to work in a folder other than fastai_audio, you can link the audio folder by navigating in the terminal to the folder where your notebooks are and running
`ln -s /path-to-fastai-audio/audio .`

This will create a symbolic link to the audio folder which contains our code. It will appear as a folder called 'audio' in your directory and you'll be able to access it using `from audio import *`, but it won't take up any additional space. To remove, just navigate to the folder where you created the symlink and type `unlink audio`. This will remove the linked folder but leave the original intact.

# Features

### Audio Transform Pre-Processors

We provide a way to resample, remove silence and segment your items before generating spectrograms.

These operations are cached and depend on each other in the order below. If you change your segment settings, you won't need to resample etc.

- Resample: e.g. 44kHz to 16KHz
- Remove Silence: Options to trim silence, split by silence, and remove all silence.
- Segment: Chop up along clip into segments e.g. 12s sample into 5s segments = [5s, 5s, 2s]

### Traintime Features

- Realtime Spectrogram Generation
- Spectrogram Caching
- Data Augmentation for Spectrograms (SpecAugment, rolling, size changes)
- MFCC (Mel-Frequency Cepstral Coefficient) Generation
- Option to append delta/accelerate 
- and much more...

# Tutorials

- 00-Getting Started - Shows basic functionality of the library and how to train a simple audio model
- 01-Intro to Audio - A detailed intro guide to the basics of audio processing, librosa, and spectrograms. Not ML-specific and doesn't use this library
- 02-Features - A detailed walk through all the libraries features.
- 03-ESC-50 - Our first real audio model, getting a new state-of-the-art on an Environmental Sound Classification problem using melspectrograms, mixup, and a simple setup.
- 04-Freesound Kaggle - A guide to using the library and it's features for the Kaggle Freesound 2018 competition on acoustic scene classification. Also uses melspectrograms and mixup and includes inference on a test set.
- Coming Soon: 05a-Googlespeech Kaggle MFCC+Delta - Using MFCC's with delta/accelerate stacking to enter the Google Tensorflow speech challenge from 2018. Includes semisupervised learning (using a model to pseudolabel an unlabeled set).
- Coming Soon: 05b-Googlespeech Kaggle Melspec Ensemble - An alternate model that uses melspectrograms and SpecAugment (no mixup). We then ensemble this with the model from 05a to do inference on a test set and submit.

# Known Issues
- We don't currently understand normalization for audio and the best way to implement it. 
- Inference and Exporting models work in some cases but are broken in others, check your outputs before using.
- Stats method can be extremely slow.
- Cache folder can get extremely large. Remember to clear it using `config.clear_cache()` or by manually removing your cache folders. 

# Contributors
We are looking for contributors of all skill levels. If you don't have time to contribute, please at least reach out and give us some feedback on the library by posting in the [fastai audio thread](https://forums.fast.ai/t/deep-learning-with-audio-thread/38123) or contact us via PM [@baz](https://forums.fast.ai/u/baz/) or [@madeupmasters](https://forums.fast.ai/u/MadeUpMasters/)

### Active Contributors
- [kevinbird15](https://github.com/kevinbird15)
- [mogwai](https://github.com/mogwai)
- [rbracco](https://github.com/rbracco)

### Past Contributors
- [artste](https://github.com/artste)
- [zcaceres](https://github.com/zcaceres)
- [thommackey](https://github.com/thommackey)