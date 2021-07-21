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

    def test_power_set(self):
        item1 = Item(1.0, 1, 1, 0) 
        item2 = Item(1.0, 1, 2, 0)
        items = [item1, item2]
        
        power_set = generate_item_power_set(items)
        power_set = list(power_set)
        self.assertTrue( len(power_set) == 4 )

    def test_construct_constrain_matrix(self):
        item1 = Item(1, 2, 1, 0) 
        item2 = Item(2, 2, 2, 0)
        items = [item1, item2]
        self.assertTrue( construct_constrain_matrix(generate_item_power_set(items), 
        2, items).shape == (2,8))
        # print(construct_probability_simplex_matrix(generate_item_power_set(items), 5))
        # print(construct_probability_simplex_matrix(generate_item_power_set(items), 5).shape)
        # print(find_optimal_val(items, 5))

        
        # heavy_items, light_items = assign_purchase_probabilities(items, 0.5)
        # print(simulate(heavy_items, light_items, 2))
        # print("here")
        # for bundle in generate_item_power_set(items):
        # #     print(bundle)
        # #     print("expected revenue: " + str(calculate_expected_revenue(bundle)))
        #     q0_bundle = bisection(f, 0, 1, 10000, bundle)
        #     for item in items:
        #         print("item: " + str(item))
        #         print("q_i:" + str(solve_V(q0_bundle * math.e ** (item.quality - 1))))
    def test_linear_program(self):
        item1 = Item(1, 2, 1, 0) 
        item2 = Item(2, 2, 2, 0)
        items = [item1, item2]
        self.assertTrue( abs(-find_optimal_val(items, 2).fun - 2.1280097) < tol)
        
if __name__ == '__main__':
    unittest.main()   