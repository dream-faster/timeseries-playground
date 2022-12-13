import pandas as pd
from models.base import Model
from all_types import ModelOverTime, TransformationsOverTime, X, y
from tqdm import tqdm
from copy import deepcopy
from utils.splitters import Splitter, Split


def walk_forward_train(
    model: Model,
    X: X,
    y: y,
    splitter: Splitter,
    transformations_over_time: TransformationsOverTime,
) -> ModelOverTime:

    # easy to parallize this with ray
    models = [
        __train_on_window(split, X, y, model, transformations_over_time)
        for split in tqdm(splitter.splits())
    ]

    idx, values = zip(*models)
    return pd.Series(values, idx).rename(model.name)


def __train_on_window(
    split: Split,
    X: X,
    y: y,
    model: Model,
    transformations_over_time: TransformationsOverTime,
) -> tuple[int, Model]:
    X_train = X.iloc[split.train_window_start : split.train_window_end].to_numpy()
    y_train = y.iloc[split.train_window_start : split.train_window_end].to_numpy()

    current_transformations = [
        transformation_over_time[split.model_index]
        for transformation_over_time in transformations_over_time
    ]

    for transformation in current_transformations:
        X_train = transformation.transform(X_train)

    current_model = deepcopy(model)
    current_model.fit(X_train, y_train)
    return split.model_index, current_model
