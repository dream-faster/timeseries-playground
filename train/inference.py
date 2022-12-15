import pandas as pd
from all_types import (
    ModelOverTime,
    TransformationsOverTime,
    InSamplePredictions,
    OutSamplePredictions,
)
from tqdm import tqdm
from models.base import Model, ModelType
from utils.flatten import single_flatten
from utils.splitters import Splitter, Split
from typing import Optional, Tuple
from utils.pandas import shift_and_duplicate_first_value
import numpy as np


def walk_forward_inference(
    model_over_time: ModelOverTime,
    transformations_over_time: Optional[TransformationsOverTime],
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Splitter,
) -> tuple[InSamplePredictions, OutSamplePredictions]:

    model: Model = model_over_time[model_over_time.index[0]]
    if model.type == ModelType.Univariate:
        X = shift_and_duplicate_first_value(y, 1)

    results = [
        __inference_from_window(
            split,
            X,
            model_over_time,
            transformations_over_time,
        )
        for split in tqdm(splitter.splits())
    ]
    results = single_flatten(results)

    idx, insample_values, outofsample_values = zip(*results)
    idx = X.index[idx[0] : idx[-1] + 1]

    insample_predictions = pd.Series(insample_values, idx).rename(
        f"{model.name}_insample_predictions"
    )
    outofsample_predictions = pd.Series(outofsample_values, idx).rename(
        f"{model.name}_outofsample_predictions"
    )
    return insample_predictions, outofsample_predictions


def __inference_from_window(
    split: Split,
    X: pd.DataFrame,
    model_over_time: ModelOverTime,
    transformations_over_time: Optional[TransformationsOverTime],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model: Model = model_over_time[split.model_index]
    X_train = X.iloc[split.train_window_start : split.train_window_end]

    if transformations_over_time is not None:
        current_transformations = [
            transformation_over_time.iloc[split.model_index]
            for transformation_over_time in transformations_over_time
        ]
        for transformation in current_transformations:
            X_train = transformation.transform(X_train)
    X_train = X_train.to_numpy()

    X_test = X.iloc[split.test_window_start : split.test_window_end]
    if transformations_over_time is not None:
        for transformation in current_transformations:
            X_test = transformation.transform(X_test)
    X_test = X_test.to_numpy()

    predictions_insample = model.predict_in_sample(X_train)
    predictions_outofsample = model.predict(X_test)
    idx = np.arange(
        split.test_window_start,
        split.test_window_start + len(predictions_outofsample),
        1,
    )

    return zip(idx, predictions_insample, predictions_outofsample)
