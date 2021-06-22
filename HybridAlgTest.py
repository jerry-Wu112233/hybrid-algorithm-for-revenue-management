import unittest
from HybridAlg import *


tol = 0.000001

iterations = 10000

class HybridAlgTest(unittest.TestCase):
 
    def test_solve_q0(self):
        item1 = Item(1.0, 1, 1, 0) 
        item2 = Item(1.0, 1, 2, 0)

        item1_q0 = bisection(f, 0, 1, iterations, [item1])
        item2_q0 = bisection(f, 0, 1, iterations, [item2]) 
        self.assertTrue( abs(item1_q0 - 0.6381037 ) < tol ) 
        self.assertTrue( abs(item2_q0 - 0.5) < tol)

    def test_solve_bundle_q0(self):
        item1 = Item(1.0, 1, 1, 0) 
        item2 = Item(1.0, 1, 2, 0)
        items = [item1, item2]

        items_q0 = bisection(f, 0, 1, iterations, items)
        self.assertTrue( f(items_q0, items) < tol )

    

if __name__ == '__main__':
    unittest.main()   