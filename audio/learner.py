from .data import *

def audio_learner(data:DataBunch, base_arch:Callable=models.resnet18, metrics=accuracy, **kwargs):
    '''Wrapper function for fastai learner. Converts head of model to fit one channel input.'''
    learn = cnn_learner(data, base_arch, metrics=metrics, **kwargs)
    newlayer = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    newlayer = newlayer.cuda() # Our layer should use cuda, since the rest of the model will.
    learn.model[0][0] = newlayer
    learn.unfreeze()
    return learn

def audio_predict(learn, item:AudioItem):
    '''Applies the AudioTransforms to the item before predicting its class'''
    config = learn.data.x.config
    path = learn.data.x.path
    al = AudioList([item], path, config=config).split_none().label_empty()
    preds = [learn.predict(x)[2] for x in al.x]
    preds = [learn.data.y.classes[(pred==torch.max(pred)).nonzero()] for pred in preds] 
    return preds
    # res = [learn.predict(ai)[2] for ai in al.x]
    # return learn.data.y.classes[torch.max(res)]
    