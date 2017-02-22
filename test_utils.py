import unittest
from utils import RangedNum, RangedInt


class RangedNumTest(unittest.TestCase):

    def setUp(self):
        self.min = 1
        self.max = 10
        self.ranged_num = RangedNum(self.min, self.max)

    def test_init(self):
        self.assertTrue(self.min <= self.ranged_num.value <= self.max)

    def test_add(self):
        new_r = self.ranged_num + 5
        self.assertIsInstance(new_r, RangedNum)
        self.assertTrue(self.min <= new_r.value <= self.max)

    def test_sub(self):
        new_r = self.ranged_num - 5
        self.assertIsInstance(new_r, RangedNum)
        self.assertTrue(self.min <= new_r.value <= self.max)

    def test_iadd(self):
        self.ranged_num += 5
        self.assertIsInstance(self.ranged_num, RangedNum)
        self.assertTrue(self.min <= self.ranged_num.value <= self.max)

    def test_isub(self):
        self.ranged_num -= 5
        self.assertIsInstance(self.ranged_num, RangedNum)
        self.assertTrue(self.min <= self.ranged_num.value <= self.max)


class RangedIntTest(unittest.TestCase):

    def setUp(self):
        self.min = 1
        self.max = 10
        self.ranged_int = RangedInt(self.min, self.max)

    def test_init(self):
        self.assertIsInstance(self.ranged_int.value, int)
        self.assertIn(self.ranged_int.value, (i for i in range(self.min, self.max+1)))

    def test_add(self):
        new_r = self.ranged_int + 5
        self.assertIsInstance(new_r, RangedInt)
        self.assertIsInstance(new_r.value, int)
        self.assertIn(new_r.value, (i for i in range(self.min, self.max+1)))

    def test_sub(self):
        new_r = self.ranged_int - 5
        self.assertIsInstance(new_r, RangedInt)
        self.assertIsInstance(new_r.value, int)
        self.assertIn(new_r.value, (i for i in range(self.min, self.max+1)))

    def test_iadd(self):
        self.ranged_int += 5
        self.assertIsInstance(self.ranged_int, RangedInt)
        self.assertIsInstance(self.ranged_int.value, int)
        self.assertIn(self.ranged_int.value, (i for i in range(self.min, self.max+1)))

    def test_isub(self):
        self.ranged_int -= 5
        self.assertIsInstance(self.ranged_int, RangedInt)
        self.assertIsInstance(self.ranged_int.value, int)
        self.assertIn(self.ranged_int.value, (i for i in range(self.min, self.max+1)))
