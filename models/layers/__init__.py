from shape import RepeatLayer
from sampling import (
    GaussianSampleLayer,
    GaussianMultiSampleLayer,
    BernoulliSampleLayer,
    GumbelSoftmaxSampleLayer,
    GumbelSampleLayer,
    LogisticSampleLayer
)
from conv import Deconv2DLayer
from misc import (
  ArgmaxLayer, RoundLayer,
  SoftmaxLayer, SigmoidLayer
)