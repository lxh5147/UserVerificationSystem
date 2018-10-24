import unittest

import numpy as np

from predict import _sim


class PredictTestCase(unittest.TestCase):
    def test_sim(self):
        embeddings = np.asanyarray([[-1., 2.], [3., 4.]], dtype=np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        sim = _sim(embeddings[0], embeddings[1])

        self.assertTrue(abs(sim - 0.447) < 0.001, 'sim')


if __name__ == '__main__':
    unittest.main()
