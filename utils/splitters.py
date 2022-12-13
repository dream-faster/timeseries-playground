from dataclasses import dataclass
from collections import Iterator
from typing import List, Optional


@dataclass
class Split:
    model_index: int
    train_window_start: int
    train_window_end: int
    test_window_start: int
    test_window_end: int


class Splitter:
    def splits(self) -> List[Split]:
        raise NotImplementedError


class SlidingWindowSplitter(Splitter):

    iterator: Iterator

    def __init__(
        self,
        start: int,
        end: int,
        window_size: int,
        step: int,
    ) -> None:
        self.window_size = window_size
        self.step = step
        self.start = start
        self.end = end
        self.iterator = iter(range(start, end, step))

    def splits(self) -> List[Split]:
        return [
            Split(
                model_index=index,
                train_window_start=index - self.window_size,
                train_window_end=index - 1,
                test_window_start=index,
                test_window_end=min(self.end - 1, index - 1 + self.step),
            )
            for index in range(self.start + self.window_size, self.end, self.step)
        ]


class ExpandingWindowSplitter(Splitter):

    iterator: Iterator

    def __init__(
        self,
        start: int,
        end: int,
        window_size: int,
        step: int,
    ) -> None:
        self.window_size = window_size
        self.step = step
        self.start = start
        self.end = end
        self.iterator = iter(range(start, end, step))

    def splits(self) -> List[Split]:
        return [
            Split(
                model_index=index,
                train_window_start=self.start,
                train_window_end=index - 1,
                test_window_start=index,
                test_window_end=min(self.end - 1, index - 1 + self.step),
            )
            for index in range(self.start + self.window_size, self.end, self.step)
        ]
