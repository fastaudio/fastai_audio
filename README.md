# Fast AI Audio

Using audio with fastai. Mainly with the spectrogram approach but optionally raw signals as well.

Run `./install.sh` to get started.

Features:

- Spectrogram Caching
- Resampling
- Removing Silence
- Segementing Samples

# Spectrogram Caching

You will see a ~20% improvement in epoch training speed after the first epoch. 

# Audio Transform Pre-Processors

We provide a way to resample, remove silence and segment your items before generating spectrograms.

These operations are cached and depend on each other in the order above. If you change your segment settings, you won't need to resample etc.

- Resample: 44kHz to 16KHz
- Remove Silence: Cut your samples into segments where the silence is removed.
- Segment: Chop up into segments e.g. 12s sample into 5s segments = [5s, 5s, 2s]

### Known Issues
- Exporting models hasn't been tested and configured yet
- Your cache folder will eat your hard drive you are messing around with a lot of different pre-transform configurations. Remember to clear it

### Contributors

- [rbracco](https://github.com/rbracco)
- [artste](https://github.com/artste)
- [zcaceres](https://github.com/zcaceres)
- [thommackey](https://github.com/thommackey)