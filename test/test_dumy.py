import unittest


def sum(arg):
    total = 0
    for val in arg:
        total += val
    return total


class TestSum(unittest.TestCase):

    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)
