from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.resnext import ResNeXt50, ResNeXt101
from models.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
from models.efficientnet import EfficientNetB5, EfficientNetB6, EfficientNetB7, EfficientNetL2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import VGG16, VGG19
from models.senet import SENet154, SEResNet50, SEResNet101, SEResNet152, SEResNeXt50, SEResNeXt101
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import MobileNet, MobileNetV2
from models.mobilenetv3 import MobileNetv3_large, MobileNetv3_small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Activation
from tensorflow.keras.models import Model


def resnet(name, size, classes, weights='imagenet', include_top=False):
    if name == 'resnet18':
        model = ResNet18((size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    elif name == 'resnet34':
        model = ResNet34((size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    elif name == 'resnet50':
        model = ResNet50((size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    elif name == 'resnet101':
        model = ResNet101((size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    elif name == 'resnet152':
        model = ResNet152((size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    else:
        raise TypeError('Unsupported model type!')

    return model


def resnext(name, size, classes, weights='imagenet', include_top=False):
    if name == 'resnext50':
        model = ResNeXt50((size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    elif name == 'resnext101':
        model = ResNeXt101((size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    else:
        raise TypeError('Unsupported model type!')

    return model


def efficientnet(name, size, classes, weights='imagenet', include_top=False):
    if name == 'efficientnetb0':
        model = EfficientNetB0(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                               classes=classes)
    elif name == 'efficientnetb1':
        model = EfficientNetB1(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                               classes=classes)
    elif name == 'efficientnetb2':
        model = EfficientNetB2(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                               classes=classes)
    elif name == 'efficientnetb3':
        model = EfficientNetB3(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                               classes=classes)
    elif name == 'efficientnetb4':
        model = EfficientNetB4(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                               classes=classes)
    elif name == 'efficientnetb5':
        model = EfficientNetB5(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                               classes=classes)
    elif name == 'efficientnetb6':
        model = EfficientNetB6(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                               classes=classes)
    elif name == 'efficientnetb7':
        model = EfficientNetB7(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                               classes=classes)
    elif name == 'efficientnetl2':
        model = EfficientNetL2(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                               classes=classes)
    else:
        raise TypeError('Unsupported model type!')

    return model


def inception(name, size, classes, weights='imagenet', include_top=False):
    if name == 'inception_resnet_v2':
        model = InceptionResNetV2(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                                  classes=classes)
    elif name == 'inceptionv3':
        model = InceptionV3(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                            classes=classes)
    else:
        raise TypeError('Unsupported model type!')

    return model


def densenet(name, size, classes, weights='imagenet', include_top=False):
    if name == 'densenet121':
        model = DenseNet121(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                            classes=classes)
    elif name == 'densenet169':
        model = DenseNet169(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                            classes=classes)
    elif name == 'densenet201':
        model = DenseNet201(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3),
                            classes=classes)
    else:
        raise TypeError('Unsupported model type!')

    return model


def vggnet(name, size, classes, weights='imagenet', include_top=False):
    if name == 'vgg16':
        model = VGG16(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3), classes=classes)
    elif name == 'vgg19':
        model = VGG19(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3), classes=classes)
    else:
        raise TypeError('Unsupported model type!')

    return model


def senet(name, size, classes, weights='imagenet', include_top=False):
    if name == 'senet154':
        model = SENet154(input_shape=(size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    elif name == 'seresnet50':
        model = SEResNet50(input_shape=(size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    elif name == 'seresnet101':
        model = SEResNet101(input_shape=(size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    elif name == 'seresnet152':
        model = SEResNet152(input_shape=(size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    elif name == 'seresnext50':
        model = SEResNeXt50(input_shape=(size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    elif name == 'seresnet101':
        model = SEResNeXt101(input_shape=(size[0], size[1], 3), weights=weights, classes=classes, include_top=include_top)
    else:
        raise TypeError('Unsupported model type!')

    return model


def xception(name, size, classes, weights='imagenet', include_top=False):
    if name == 'xception':
        model = Xception(include_top=include_top, weights=weights, input_shape=(size[0], size[1], 3), classes=classes)
    else:
        raise TypeError('Unsupported model type!')

    return model


def mobilenet(name, size, classes, weights='imagenet', include_top=False):
    if name == 'mobilenetv1':
        model = MobileNet(input_shape=(size[0], size[1], 3), include_top=include_top, weights=weights,
                          classes=classes)
    elif name == 'mobilenetv2':
        model = MobileNetV2(input_shape=(size[0], size[1], 3), include_top=include_top, weights=weights,
                            classes=classes)
    elif name == 'mobilenetv3l':
        model = MobileNetv3_large(classes=classes, input_shape=(size[0], size[1], 3), include_top=include_top)
    elif name == 'mobilenetv3m':
        model = MobileNetv3_small(classes=classes, input_shape=(size[0], size[1], 3), include_top=include_top)
    else:
        raise TypeError('Unsupported model type!')

    return model


def nasnet(name, size, classes, weights='imagenet', include_top=False):
    if name == 'nasnetlarge':
        model = NASNetLarge(input_shape=(size[0], size[1], 3), include_top=include_top, weights=weights,
                            classes=classes)
    elif name == 'nasnetmobile':
        model = NASNetMobile(input_shape=(size[0], size[1], 3), include_top=include_top, weights=weights,
                             classes=classes)
    else:
        raise TypeError('Unsupported model type!')

    return model


def get_model(name, cfg):
    size = cfg.DATASET.SIZE
    classes = cfg.DATASET.N_CLASSES
    weights = cfg.TRAIN.WEIGHTS
    include_top = False
    activation = cfg.TRAIN.ACTIVATION

    if name.startswith('resnet'):
        backbone = resnet(name, size, classes, weights, include_top)
    elif name.startswith('resnext'):
        backbone = resnext(name, size, classes, weights, include_top)
    elif name.startswith('efficientnet'):
        backbone = efficientnet(name, size, classes, weights, include_top)
    elif name.startswith('inception'):
        backbone = inception(name, size, classes, weights, include_top)
    elif name.startswith('densenet'):
        backbone = densenet(name, size, classes, weights, include_top)
    elif name.startswith('vggnet'):
        backbone = vggnet(name, size, classes, weights, include_top)
    elif name.startswith('se'):
        backbone = senet(name, size, classes, weights, include_top)
    elif name.startswith('xception'):
        backbone = xception(name, size, classes, weights, include_top)
    elif name.startswith('mobilenet'):
        backbone = mobilenet(name, size, classes, weights, include_top)
    elif name.startswith('nasnet'):
        backbone = nasnet(name, size, classes, weights, include_top)
    else:
        raise TypeError('Unsupported model type!')
    if include_top:
        return backbone
    inputs = backbone.input
    x = backbone.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(units=classes, name='logits')(x)
    if activation in {'softmax', 'sigmoid'}:
        x = Activation(activation=activation)(x)
    return Model(inputs, x, name=name)


if __name__ == '__main__':
    model1 = NASNetMobile(input_shape=(320, 320, 3), include_top=True, weights=None,
                        classes=3)
    model1.summary()
