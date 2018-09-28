import unittest

from evaluate import _verfication_fa_fr, _eer, _verification_eer


def almost_equal(a, b, es=0.0001):
    return abs(a - b) < es


class EvaluateTestCase(unittest.TestCase):
    def test_verfication_fa_fr(self):
        to_be_verified = [(0, 1), (1, 1), (2, 0)]
        sims = [0.6, 0.7, 0.8]
        true_a = [0, 2]
        true_r = [1]
        threshold = 0.7  # embedding 0 is rejected --> false rejection, 1 is accepted --> false acceptance, 2 is accepted
        fa, fr = _verfication_fa_fr(sims, true_a, true_r, threshold)
        self.assertEqual(sorted(fa), [1], 'fa')
        self.assertEqual(sorted(fr), [0], 'fr')

    def test_eer(self):
        fa_rates = [0.05, 0.10, 0.12]
        fr_rates = [0.20, 0.15, 0.13]
        error_rates = [0.15, 0.20, 0.22]
        thresholds = [0.1, 0.2, 0.3]
        fa_rate, fr_rate, error_rate, threshold = _eer(fa_rates, fr_rates, error_rates, thresholds)
        self.assertEqual(fa_rate, 0.12, 'false acceptance rate')
        self.assertEqual(fr_rate, 0.13, 'false rejection rate')
        self.assertEqual(error_rate, 0.22, 'error rate')
        self.assertEqual(threshold, 0.3, 'threshold')

    def test_verification_eer(self):
        sims = [0.6, 0.7, 0.8]
        true_a = [0, 2]
        true_r = [1]
        fa_rate, fr_rate, error_rate, threshold = _verification_eer( sims, true_a, true_r)
        self.assertEqual(fa_rate, 1.0, 'false acceptance rate')
        self.assertEqual(fr_rate, 0.5, 'false rejection rate')
        self.assertTrue(almost_equal(error_rate, 0.66666), 'error rate')
        self.assertTrue(almost_equal(threshold, 0.6), 'threhold')


if __name__ == '__main__':
    unittest.main()
