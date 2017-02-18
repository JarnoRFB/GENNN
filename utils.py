import random


class RangedInt:


    def __init__(self, min, max, value="random"):

        self.min = min
        self.max = max
        if value == "random":
            self.value = random.randint(self.min, self.max)

    def __add__(self, other):
        return max(self.min, min(self.max, self.value + other))

    def __sub__(self, other):
        return max(self.min, min(self.max, self.value - other))

    def __eq__(self, other):
        return (self.value == other)

    @property
    def value(self):
        return self.value

    @value.setter
    def value(self, value):
        self.value = max(self.min, min(value, self.max))
