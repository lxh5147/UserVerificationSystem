import unittest

from evaluate import _verfication_fa_fr


class EvaluateTestCase(unittest.TestCase):
    def test_verfication_fa_fr(self):
        to_be_verified = [(0, 1), (1, 1), (2, 0)]
        sims = [0.6, 0.7, 0.8]
        true_a = [0, 2]
        true_r = [1]
        threshold = 0.7  # embedding 0 is rejected --> false rejection, 1 is accepted --> false acceptance, 2 is accepted
        fa, fr = _verfication_fa_fr(to_be_verified, sims, true_a, true_r, threshold)
        self.assertEqual(sorted(fa), [1], 'fa')
        self.assertEqual(sorted(fr), [0], 'fr')


if __name__ == '__main__':
    unittest.main()
