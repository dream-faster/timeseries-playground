import pandas as pd
from all_types import (
    X,
    ModelOverTime,
    TransformationsOverTime,
    InSamplePredictions,
    OutSamplePredictions,
)
from tqdm import tqdm
from models.base import Model

from utils.splitters import Splitter, Split


def walk_forward_inference(
    model_over_time: ModelOverTime,
    transformations_over_time: TransformationsOverTime,
    X: X,
    splitter: Splitter,
) -> tuple[InSamplePredictions, OutSamplePredictions]:
    insample_predictions = pd.Series(index=X.index, dtype="object").rename(
        f"insample_predictions"
    )
    outofsample_predictions = pd.Series(index=X.index, dtype="object").rename(
        f"outofsample_predictions"
    )

    batched_results = [
        __inference_from_window(
            split,
            X,
            model_over_time,
            transformations_over_time,
        )
        for split in tqdm(splitter.splits())
    ]
    for batch in batched_results:
        for index, prediction, probs in batch:
            insample_predictions.iloc[index] = prediction
            outofsample_predictions.iloc[index] = probs

    return insample_predictions, outofsample_predictions


def __inference_from_window(
    split: Split,
    X: X,
    model_over_time: ModelOverTime,
    transformations_over_time: TransformationsOverTime,
) -> list[tuple[int, float, pd.Series]]:
    current_model: Model = model_over_time.iloc[split.model_index]
    current_transformations = [
        transformation_over_time.iloc[split.model_index]
        for transformation_over_time in transformations_over_time
    ]

    X_test = X.iloc[split.test_window_start : split.test_window_end]

    for transformation in current_transformations:
        X_test = transformation.transform(X_test)

    X_test = X_test.to_numpy()

    predictions_insample = current_model.predict_in_sample(X_test)
    predictions_outofsample = current_model.predict(X_test)
    results = [
        (
            split.test_window_start + index,
            predictions_insample[index],
            predictions_outofsample[index],
        )
        for index in range(len(predictions_outofsample))
    ]

    return results
