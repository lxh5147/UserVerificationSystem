import unittest
import numpy as np
from predict import l2_norm

class PredictTestCase(unittest.TestCase):
    def test_l2_norm(self):
        embeddings=np.asanyarray([[1.,2.],[3.,4.]], dtype=np.float32)
        normed_embeddings = l2_norm(embeddings)
        norm = np.linalg.norm(normed_embeddings, ord=2, axis=1)
        self.assertTrue((np.abs(norm-1.0)<0.00001).all(),'l2_norm')

if __name__ == '__main__':
    unittest.main()
