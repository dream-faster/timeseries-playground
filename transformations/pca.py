from __future__ import annotations
from .base import Transformation
from typing import Optional
from copy import deepcopy
from sklearn.decomposition import PCA
import pandas as pd


class PCATransformation(Transformation):

    pca: PCA

    def __init__(self, ratio_components_to_keep: float, initial_window_size: int):
        self.ratio_components_to_keep = ratio_components_to_keep
        self.initial_window_size = initial_window_size

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        self.pca = PCA(
            n_components=min(
                int(len(X.columns) * self.ratio_components_to_keep),
                self.initial_window_size,
            ),
            # whiten=True,
        )
        self.pca.fit(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(self.pca.transform(X), index=X.index)
        X.columns = ["PCA_" + str(i) for i in range(1, len(X.columns) + 1)]
        return X

    def clone(self) -> PCATransformation:
        return deepcopy(self)
