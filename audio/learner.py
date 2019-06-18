from .data import *

def audio_predict(learn, item:AudioItem):
    '''Applies the AudioTransforms to the item before predicting its class'''
    config = learn.data.x.config
    path = learn.data.x.path
    al = AudioList([item], path, config=config).split_none().label_empty()
    res = torch.tensor([learn.predict(ai)[1] for ai in al.x])
    return learn.data.y.classes[torch.max(res)]
    