import random
from json import JSONEncoder

class RangedNum:
    """Keeps a numeric value in a certain range."""

    def __init__(self, min, max, value="random"):

        self._min = min
        self._max = max
        if value == "random":
            self._value = random.uniform(self._min, self._max)
        else:
            self._value = value

    def __repr__(self):
        return '<RangedNum(min={}, max={}, value={})>'.format(self._min, self._max, self.value)

    def __add__(self, other):
        return RangedNum(
            self._min,
            self._max,
            value=max(self._min, min(self._max, self._value + other))
        )

    def __sub__(self, other):
        return RangedNum(
            self._min,
            self._max,
            value=max(self._min, min(self._max, self._value - other))
        )

    def __iadd__(self, other):
        self._value = max(self._min, min(self._max, self._value + other))
        return self

    def __isub__(self, other):
        self. value = max(self._min, min(self._max, self._value - other))
        return self

    def __eq__(self, other):
        return (self._value == other)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = max(self._min, min(value, self._max))


class RangedInt:
    """Keeps an integer value in a certain range."""

    def __init__(self, min, max, value="random"):

        self._min = min
        self._max = max
        if value == "random":
            self._value = random.randint(self._min, self._max)
        else:
            self._value = round(value)

    def __repr__(self):
        return '<RangedInt(min={}, max={}, value={})>'.format(self._min, self._max, self.value)

    def __add__(self, other):
        return RangedNum(
            self._min,
            self._max,
            value=round(max(self._min, min(self._max, self._value + other)))
        )

    def __sub__(self, other):
        return RangedNum(
            self._min,
            self._max,
            value=round(max(self._min, min(self._max, self._value - other)))
        )

    def __iadd__(self, other):
        self._value = round(max(self._min, min(self._max, self._value + other)))
        return self

    def __isub__(self, other):
        self._value = round(max(self._min, min(self._max, self._value - other)))
        return self

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = max(self._min, min(value, self._max))


class RangedJSONEncoder(JSONEncoder):

    def default(self, o):

        if isinstance(o, RangedNum) or isinstance(o, RangedInt):
            return o.value
        return super().default(o)