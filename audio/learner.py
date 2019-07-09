from .data import *

#wrapper for cnn learner that does nothing, but will be used in the future for handling raw waveform (1D) inputs.
def audio_learner(data:DataBunch, base_arch:Callable=models.resnet18, metrics=accuracy, **kwargs):
    '''Wrapper function for fastai learner. Converts head of model to fit one channel input.'''
    learn = cnn_learner(data, base_arch, metrics=metrics, **kwargs)
    return learn

def audio_predict(learn, item:AudioItem):
    '''Applies the AudioTransforms to the item before predicting its class'''
    al = AudioList([item], path=learn.data.x.path, config=learn.data.x.config).split_none().label_empty()
    spectro = AudioList.open(al, item.path).spectro
    return learn.predict(spectro)

def window_predict(learn, item:AudioItem, cfg, hop):
    display(item.spectro)
    