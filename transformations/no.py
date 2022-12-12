from __future__ import annotations
from .base import Transformation
from typing import Optional
import pandas as pd


class NoTransformation(Transformation):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def clone(self) -> NoTransformation:
        return self
