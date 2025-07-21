from collections.abc import Sequence


import keras

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable, serialize

from .distribution import Distribution


@serializable("bayesflow.distributions")
class Categorical(Distribution):
    def __init__(self, prob_weights=Sequence[float], **kwargs):
        super().__init__(**kwargs)
        self.prob_weights = prob_weights
        probs = [p / sum(prob_weights) for p in prob_weights]
        self.probs = keras.ops.convert_to_tensor([probs])

    def get_config(self):
        base_config = super().get_config()

        config = {"prob_weights": self.prob_weights, "probs": self.probs}

        return base_config | serialize(config)

    def build(self, input_shape) -> None:
        self.built = True

    def log_prob(self, samples: Tensor, conditions: Tensor, **kwargs):
        # samples need to be one-hot encoded!
        prob = samples * self(conditions)
        prob = keras.ops.sum(prob, axis=0)

        return keras.ops.log(prob)

    def __call__(self, conditions: Tensor, *args, **kwargs):
        batch_size = keras.ops.shape(conditions)[0]

        return keras.ops.repeat(self.probs, repeats=batch_size, axis=0)
