from .data import *

#wrapper for cnn learner that does nothing, but will be used in the future for handling raw waveform (1D) inputs.
def audio_learner(data:DataBunch, base_arch:Callable=models.resnet18, metrics=accuracy, **kwargs):
    '''Wrapper function for fastai learner. Converts head of model to fit one channel input.'''
    learn = cnn_learner(data, base_arch, metrics=metrics, **kwargs)
    return learn

def audio_predict(learn, item:AudioItem):
    '''Applies the AudioTransforms to the item before predicting its class'''
    if(isinstance(item, torch.Tensor)): return learn.predict(item.squeeze(0).cpu())
    al = AudioList([item], path=learn.data.path, config=learn.data.x.config).split_none().label_empty()
    spectro = AudioList.open(al, item.path).spectro
    return learn.predict(spectro)

def get_audio_preds(learn, al:AudioList):
    al = al.split_none().label_empty()
    data = [AudioList.open(al, ai[0].path).spectro for ai in al.train]
    preds = [learn.predict(spectro)[1:] for spectro in progress_bar(data)]
    grouped = [o for o in zip(*preds)]
    cats = torch.cat([o.view(1) for o in grouped[0]])
    tens = torch.cat(grouped[1])
    return (cats, tens)

def window_predict(learn, item:AudioItem, cfg, hop):
    display(item.spectro)
    