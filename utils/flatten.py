from typing import List


def single_flatten(l: List) -> List:
    return [item for sublist in l for item in sublist]
