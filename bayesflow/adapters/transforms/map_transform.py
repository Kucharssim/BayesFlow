import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform
from .transform import Transform


@serializable
class MapTransform(Transform):
    """
    Implements a transform that applies a set of elementwise transforms
    to a subset of the data as given by a mapping.

    Parameters
    ----------
    transform_map : dict[str, ElementwiseTransform]
        Dictionary with variable names as keys and corresponding transforms as values.
    """

    def __init__(self, transform_map: dict[str, ElementwiseTransform]):
        self.transform_map = transform_map

    def __repr__(self):
        # if the transform map values are all the same type, use that type name
        transform_types = {type(transform) for transform in self.transform_map.values()}

        if len(transform_types) == 1:
            transform_type = transform_types.pop()
        else:
            transform_type = MapTransform

        if e := self.extra_repr():
            return f"{transform_type.__name__}({e})"

        return transform_type.__name__

    def get_config(self) -> dict:
        return serialize({"transform_map": self.transform_map})

    def forward(self, data: dict[str, np.ndarray], *, strict: bool = True, **kwargs) -> dict[str, np.ndarray]:
        data = data.copy()

        required_keys = set(self.transform_map.keys())
        available_keys = set(data.keys())
        missing_keys = required_keys - available_keys

        if strict and missing_keys:
            raise KeyError(f"Missing keys: {missing_keys!r}")

        for key, transform in self.transform_map.items():
            if key in data:
                data[key] = transform.forward(data[key], **kwargs)

        return data

    def inverse(self, data: dict[str, np.ndarray], *, strict: bool = False, **kwargs) -> dict[str, np.ndarray]:
        data = data.copy()

        required_keys = set(self.transform_map.keys())
        available_keys = set(data.keys())
        missing_keys = required_keys - available_keys

        if strict and missing_keys:
            raise KeyError(f"Missing keys: {missing_keys!r}")

        for key, transform in self.transform_map.items():
            if key in data:
                data[key] = transform.inverse(data[key], **kwargs)

        return data
