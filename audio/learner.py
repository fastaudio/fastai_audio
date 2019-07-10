from .data import *

#wrapper for cnn learner that does nothing, but will be used in the future for handling raw waveform (1D) inputs.
def audio_learner(data:DataBunch, base_arch:Callable=models.resnet18, metrics=accuracy, **kwargs):
    '''Wrapper function for fastai learner. Converts head of model to fit one channel input.'''
    return cnn_learner(data, base_arch, metrics=metrics, **kwargs)

def audio_predict(learn, item:AudioItem):
    '''Applies preprocessing to an AudioItem, or spectrogram before predicting its class'''
    if(isinstance(item, torch.Tensor)): return learn.predict(item.cpu())
    al = AudioList([item], path=item.path, config=learn.data.x.config).split_none().label_empty()
    spectro = AudioList.open(al, item.path).spectro
    return learn.predict(spectro)                                              

def audio_predict_all(learn, al:AudioList):
    '''Applies preprocessing to an AudioList then predicts on all items'''
    al = al.split_none().label_empty()
    data = [AudioList.open(al, ai[0].path).spectro for ai in al.train]
    preds = [learn.predict(spectro) for spectro in progress_bar(data)]
    return [o for o in zip(*preds)]