from __future__ import print_function, division, absolute_import
from .fbresnet import fbresnet152

from .cafferesnet import cafferesnet101

from .bninception import bninception

from .resnext import resnext101_32x4d
from .resnext import resnext101_64x4d

from .inceptionv4 import inceptionv4

from .inceptionresnetv2 import inceptionresnetv2

from .nasnet import nasnetalarge

from .nasnet_mobile import nasnetamobile

from .torchvision_models import alexnet
from .torchvision_models import densenet121
from .torchvision_models import densenet169
from .torchvision_models import densenet201
from .torchvision_models import densenet161
from .torchvision_models import resnet18
from .torchvision_models import resnet34
from .torchvision_models import resnet50
from .torchvision_models import resnet101
from .torchvision_models import resnet152
from .torchvision_models import inceptionv3
from .torchvision_models import squeezenet1_0
from .torchvision_models import squeezenet1_1
from .torchvision_models import vgg11
from .torchvision_models import vgg11_bn
from .torchvision_models import vgg13
from .torchvision_models import vgg13_bn
from .torchvision_models import vgg16
from .torchvision_models import vgg16_bn
from .torchvision_models import vgg19_bn
from .torchvision_models import vgg19

from .dpn import dpn68
from .dpn import dpn68b
from .dpn import dpn92
from .dpn import dpn98
from .dpn import dpn131
from .dpn import dpn107

from .xception import xception

from .senet import senet154
from .senet import se_resnet50
from .senet import se_resnet101
from .senet import se_resnet152
from .senet import se_resnext50_32x4d
from .senet import se_resnext101_32x4d

from .pnasnet import pnasnet5large
from .polynet import polynet

from .i3d import i3d, i3d_flow
from .resnet3D import resnet3d10
from .resnet3D import resnet3d18
from .resnet3D import resnet3d34
from .resnet3D import resnet3d50
from .resnet3D import resnet3d101
from .resnet3D import resnet3d152
from .resnet3D import resnet3d200

from .resnet3D import resneti3d18
from .resnet3D import resneti3d50
from .resnet3D import resneti3d101
from .resnet3D import resneti3d152
from .resnet3D import resneti3d200

from .resnext3D import resnext3d10
from .resnext3D import resnext3d18
from .resnext3D import resnext3d34
from .resnext3D import resnext3d50
from .resnext3D import resnext3d101
from .resnext3D import resnext3d152
from .resnext3D import resnext3d200

from .resnest import resnest18
from .resnest import resnest50
from .resnest import resnest101
from .resnest import resnest200
from .resnest import resnest269

from .slowfast import slowfast18
from .slowfast import slowfast50
from .slowfast import slowfast101
from .slowfast import slowfast152
from .slowfast import slowfast200

from .mxresnet import mxresnet18
from .mxresnet import mxresnet34
from .mxresnet import mxresnet50
from .mxresnet import mxresnet50
from .mxresnet import mxresnet101
from .mxresnet import mxresnet152
from .mxresnet import samxresnet18
from .mxresnet import samxresnet34
from .mxresnet import samxresnet50
from .mxresnet import samxresnet50
from .mxresnet import samxresnet101
from .mxresnet import samxresnet152
from .mxresnet import ssamxresnet18
from .mxresnet import ssamxresnet34
from .mxresnet import ssamxresnet50
from .mxresnet import ssamxresnet50
from .mxresnet import ssamxresnet101
from .mxresnet import ssamxresnet152

from .simclr_resnet import resnet50x1 as simclr_resnet50x1
from .simclr_resnet import resnet50x2 as simclr_resnet50x2
from .simclr_resnet import resnet50x4 as simclr_resnet50x4

from .moco import resnet50 as moco_resnet50

from .mobilenet import mobilenetv2

from .soundnet import soundnet8
from .memnet import memnet

from .utils import Identity

from .detection import facenet
