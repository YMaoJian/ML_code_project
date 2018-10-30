from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mxnet.initializer import Xavier

vgg_spec = {11: ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
            13: ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
            16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
            19: ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])}
            
class VGG(HybridBlock):
    def __init__(self, layers, filters, classes=1000, batch_norm=False, **kwargs):
        super(VGG, self).__init__(**kwargs)
        with self.name_scope():
            self.features = self._make_features(layers, filters, batch_norm)
            self.features.add(nn.Dense(4096, activation='relu',
                                       weight_initializer='normal',
                                       bias_initializer='zeros'))
            self.features.add(nn.Dropout(rate=0.5))
            self.features.add(nn.Dense(4096, activation='relu',
                                       weight_initializer='normal',
                                       bias_initializer='zeros'))
            self.features.add(nn.Dropout(rate=0.5))
            self.output = nn.Dense(classes, 
                                   weight_initializer='normal',
                                   bias_initializer='zeros')
            
    def _make_features(self, layers, filters, batch_norm):
        featurizer = nn.HybridSequential(prefix='')
        for i, num in enumerate(layers):
            for _ in range(num):
                featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1, 
                                         weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                         bias_initializer='zeros'))
                if batch_norm:
                    featurizer.add(nn.BatchNorm())
                featurizer.add(nn.Activation('relu'))
            featurizer.add(nn.MaxPool2D(strides=2))
            return featurizer
        
    def hybri_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x
    
def get_vgg(num_layers, **kwargs):
    layers, filters = vgg_spec[num_layers]
    net = VGG(layers, filters, **kwargs)
    
def vgg11(**kwargs):
    return get_vgg(11, **kwargs)
    
def vgg13(**kwargs):
    return get_vgg(13, **kwargs)
    
def vgg16(**kwargs):
    return get_vgg(16, **kwargs)
    
def vgg19(**kwargs):
    return get_vgg(19, **kwargs)
