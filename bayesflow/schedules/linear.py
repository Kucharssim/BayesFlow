import keras
from bayesflow.utils.serialization import serializable, serialize


@serializable("bayesflow.schedules")
class Linear(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start: float, slope: float, min: float = 0, max: float = 1, dtype: str = "float32"):
        super().__init__()
        self.start = start
        self.min = min
        self.max = max
        self.slope = slope
        self.dtype = dtype

    def __call__(self, step):
        x = self.start + self.slope * keras.ops.cast(step, self.dtype)
        return keras.ops.clip(x, x_min=self.min, x_max=self.max)

    def get_config(self):
        config = {"start": self.start, "min": self.min, "max": self.max, "slope": self.slope, "dtype": self.dtype}

        return serialize(config)
