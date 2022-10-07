"""Implementation of MILAN.

MILAN consists of an encoder network based on a pretrained image classifier,
a decoder to compute p(description | image regions) that is based on an
attentional LSTM, and a plain LSTM language model to compute p(description).
The root model is the decoder, which contains the encoder and the language
model LSTM as children in addition to the attentional LSTM.

To use a pretrained model, see the `pretrained` function in `loaders.py`.
To train your own MILAN model, see `scripts/train_milan.py`.
"""
# flake8: noqa
from utils.milan.milan.decoders import Decoder, decoder
from utils.milan.milan.encoders import (Encoder, PyramidConvEncoder,
                                SpatialConvEncoder, encoder)
from utils.milan.milan.lms import LanguageModel, lm
from utils.milan.milan.loaders import pretrained
