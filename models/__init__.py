from .slowfast import MySlowFast
from .timesformer import MyTimeSformer
from .rostfine import RoSTFine
from .R3D import MyR3D
from .X3D import MyX3D
from .R21D import MyR21D
from .vivit import MyViViT
from .I3D import MyI3D
from .VGG16 import MyVgg

def model_select(args):
    # TODO: add models
    model_factory = {
        'vgg': MyVgg(args),
        'r3d': MyR3D(args),
        'r21d': MyR21D(args),
        'x3d': MyX3D(args),
        'i3d': MyI3D(args),
        'slowfast': MySlowFast(args),
        'vivit': MyViViT(args),
        'timesformer': MyTimeSformer(args),
        'rostfine': RoSTFine(args)
    }
    return model_factory[args.model_name]