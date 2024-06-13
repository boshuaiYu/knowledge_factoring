# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .hugging_face import HuggingFaceClassifier
from .image import ImageClassifier
from .timm import TimmClassifier
from .kf import KFImageClassifier

__all__ = [
    'BaseClassifier', 'ImageClassifier', 'TimmClassifier',
    'HuggingFaceClassifier', 'KFImageClassifier'
]
