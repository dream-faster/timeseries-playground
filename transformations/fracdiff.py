from __future__ import annotations
from .base import Transformation
from typing import Optional
from copy import deepcopy
from fracdiff.sklearn import FracdiffStat
import pandas as pd


class FracdiffTransformation(Transformation):

    model: FracdiffStat

    def __init__(self, window_size: int):
        self.window_size = window_size

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.model = FracdiffStat(window=self.window_size)
        self.model.fit(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(self.model.transform(X), index=X.index)
        X.columns = [f"{column}_fracdiff_{self.window_size}" for column in X.columns]
        return X

    def clone(self) -> FracdiffTransformation:
        return deepcopy(self)
