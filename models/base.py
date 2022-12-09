from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import pandas as pd
from all_types import X, y, InSamplePredictions, OutSamplePredictions


class Model(ABC):

    name: str = ""

    @abstractmethod
    def fit(self, X: X, y: y) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: X) -> OutSamplePredictions:
        raise NotImplementedError

    @abstractmethod
    def predict_in_sample(self, X: X) -> InSamplePredictions:
        raise NotImplementedError
