import random
from json import JSONEncoder


class RangedNum:
    """Keeps a numeric value in a certain range."""

    def __init__(self, minimum, maximum, value=None):

        self._min = minimum
        self._max = maximum
        if value is None:
            self.value = self._get_random()
        else:
            # Use setter to ensure that value is in specified range.
            self.value = value

    def __repr__(self):
        return '<RangedNum(min={}, max={}, value={})>'.format(self._min, self._max, self.value)

    def __add__(self, other):
        return type(self)(
            self._min,
            self._max,
            value=self._value + other
        )

    def __sub__(self, other):
        return type(self)(
            self._min,
            self._max,
            value=self._value - other
        )

    def __iadd__(self, other):
        self.value = self._value + other
        return self

    def __isub__(self, other):
        self.value = self._value - other
        return self

    def __eq__(self, other):
        return self._value == other

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = max(self._min, min(value, self._max))

    def _get_random(self):
        return random.uniform(self._min, self._max)


class RangedInt(RangedNum):
    """Keeps an integer value in a certain range."""

    def __repr__(self):
        return '<RangedInt(min={}, max={}, value={})>'.format(self._min, self._max, self.value)

    @RangedNum.value.setter
    def value(self, value):
        self._value = round(max(self._min, min(value, self._max)))

    def _get_random(self):
        return random.randint(self._min, self._max)


class RangedJSONEncoder(JSONEncoder):

    def default(self, o):

        if isinstance(o, RangedNum) or isinstance(o, RangedInt):
            return o.value
        return super().default(o)

def flip_coin(bias=0.5):
    """Flip a biased coin."""
    return random.random() <= bias