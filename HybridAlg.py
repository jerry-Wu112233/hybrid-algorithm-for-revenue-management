from dataclasses import dataclass
import numpy as np
from sympy import lambdify
from scipy.optimize import fsolve
import random
import math
from itertools import chain, combinations
from scipy.optimize import linprog
import matplotlib.pyplot as plt

@dataclass
class Item:
    '''A data class with four attributes
    
    original_inventory: a positive integer that describes the original inventory level of an item
    inventory_level: a positive integer that describes the current inventory level of an item
    quality: a real number that describes the quality of an item (could be negative or positive)
    purchase_probability: a real number in [0,1]
    '''

    original_inventory: int 
    inventory_level: int 
    quality: float 
    purchase_probability: float 
    

    def __eq__(self, o: object) -> bool:
        '''Equal method for compariing two Item type objects
        Args:
            self, o: Item object
        Returns:
            true if two Item objects have the same attribute values, false otherwise
        '''
        return ( self.original_inventory == o.original_inventory and 
               self.inventory_level == o.inventory_level and 
               self.quality == o.quality and 
               self.purchase_probability == o.purchase_probability )
       


def assign_purchase_probabilities(items, lambda_const):
    '''Assigns purchase probabilities to each item within the items iterable-object

    Computes the probability of each items with the given Assortment S := {i} for all i in items, then
    this method will sort the list of items in place based on their quality attribute. Lastly, 
    the method generated both the heavy and light list of items based on lambda_const
    
    Args:
        items: a list of Item objects
        lambda_const: a float constant in [0,1]

    Returns:
        Two lists of Item objects. the first list, heavy_items, contain all of the items with purchase probability
        that is greater than or equal to lambda_const. The second list of Items, light_items, constain all of the 
        Items with purchase probability that is less than lambda_const.
    '''
    heavy_items = []
    light_items = []

    items.sort(key = lambda x: x.quality, reverse = True)
    for item in items:
        
        item.purchase_probability = 1 - bisection(f, 0, 1, 500, [item])
        if (item.purchase_probability >= lambda_const):
            heavy_items.append(item)
        else:
            light_items.append(item)    
        
    return heavy_items, light_items
    

def solve_V(x):
    '''Solves the equation y * exp(y / (1 - y)) = x, for y. 

    Helper method for computing the monotonic and injective function V(x)

    Args:
        x: a positive real number (type: float)

    Returns:
        the solution to the equation V(x), the solution is a number within [0, 1]
    '''
    try:
        np.seterr(divide='ignore', invalid='ignore')
        func = lambda y : y * np.exp(y / (1 - y)) - x
        
    except OverflowError:
        return 1
    
    y_initial_guess = 0.5
    return fsolve(func, y_initial_guess,maxfev=30)


def bisection(f,a,b,N, light_items):
    '''Approximate solution of f(x)=0 on interval [a,b] by bisection method.
    Found from https://secure.math.ubc.ca/~pwalls/math-python/roots-optimization/bisection/
    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    a,b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    N : (positive) integer
        The number of iterations to implement.

    Returns
    -------
    x_N : number
        The midpoint of the Nth interval computed by the bisection method. The
        initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and return None.
    '''

    if len(light_items) == 0:
        return 1

    a_n = a
    b_n = b
    for _ in range(N):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n,light_items)
        if f(a_n, light_items) * f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n, light_items) * f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            return m_n
        else:
            return None

    return (a_n + b_n) / 2

def f(q0, light_items):
    '''A gateway equation method used for the bisection method

    Helper function used to compute the function f(q0) = Sum ( q0(S) * e ^ (theta_i - 1)) - (1 - q0(S)) 
    in order the find the root q0(S).

    Args:
        q0: the current estimate of the no-purchase probability given a set S of Items
        light_items: all of the Items with weight that is less than lambda

    Returns:
        The current function value of f.
    '''
    function_value = 0
    for item in light_items:
        function_value += solve_V(q0 * np.e ** (item.quality - 1))

    return function_value - 1 + q0

def update_bundle_probabilities(light_items):
    '''Finds the purchase probability when multiple items are offered in a bundle

    Calls the function that computes that no-purchase probability when a bundle of 
    items are offered. Then, using the no-purchase probability, it will update the 
    purchase_probability of each item in the bundle

    Args:
        light_items: a list of Item objects that needs to have their purchase 
        probability updated

    Returns:
        The approximate no-purchase probability associated with the bundle.
    '''
    q0_approx = bisection(f, 0, 1, 500, light_items)
    for item in light_items:
        item.purchase_probability = solve_V(q0_approx * np.e ** (item.quality - 1))
    return q0_approx

def weight_function(x):
    # if np.iscomplex((math.e / ( 1 - math.e )) * (1 - math.e)**(-x)): return 1.0
    # else:
    #     return (math.e / ( 1 - math.e )) * (1 - math.e)**(-x)
    return x**2
def assign_probability_v2(items, lambda_const):
    heavy_items = []
    light_items = []

    for item in items:
        
        item.purchase_probability = 1 - bisection(f, 0, 1, 500, [item])
        relative_weight = item.purchase_probability * weight_function(item.inventory_level / item.original_inventory)

        if relative_weight >= lambda_const:
            heavy_items.append(item)
        else:
            light_items.append(item)    
    heavy_items.sort(key = lambda x: x.purchase_probability * weight_function(x.inventory_level / x.original_inventory), reverse = True)
    
    return heavy_items, light_items

def simulate_2(items, m_const, lambda_const):
    print("beginning simulation2")
    total_revenue = 0
    sold_items = []
    for _ in range(m_const - 1):
        heavy_items, light_items = assign_probability_v2(items, lambda_const)
        if heavy_items:
            if random.uniform(0, 1) <= heavy_items[0].purchase_probability:
                heavy_items[0].inventory_level -= 1
                total_revenue += 1 / ( 1 - heavy_items[0].purchase_probability )
                if heavy_items[0].inventory_level <= 0:
                    sold_items.append(heavy_items.pop(0))
        else:                                                                       # Phase 2 of the algorithm: offering light items altogether
            if light_items:
                q0_approx = update_bundle_probabilities(light_items)
                probs = [o.purchase_probability for o in light_items]
                probs.append(q0_approx)
                for i in probs:
                    i = i / sum(probs)
                no_purchase_item = Item(0, 0, 0, q0_approx)
                light_items.append(no_purchase_item)
                item_bought = light_items[0]
                if sum(probs) != 1:
                    item_bought = np.random.choice(light_items)
                else:    
                    item_bought = np.random.choice(light_items, 1,p=probs)[0]
                    sold_items.append(item_bought)
                total_revenue += 1 / ( 1 - item_bought.purchase_probability ) 
                if item_bought == no_purchase_item:

                    light_items.remove(item_bought)
                else:
                    light_items[light_items.index(item_bought)].inventory_level -= 1
                    if light_items[light_items.index(item_bought)].inventory_level <= 0:
                        light_items.remove(item_bought)
                    light_items.remove(no_purchase_item)
        items = heavy_items + light_items
    return total_revenue

def simulate(heavy_items, light_items, m_const):
    '''Creates a simultation with m_const customers that purchases 1 inventory level
    if the Item is selected. The heavy_items are offered first, one by one. If all of 
    heavy_items are sold, then the light_items are offered altogether in a bundle.

    Args:
        heavy_items: a list of Items with purchase probability greater than lambda_const
        light_items: a list of Items with purchase probability less than lambda_const
        m_const: a positive integer describing the number of customers  
    
    Returns:
        total_revenue: sum of the prices of the Items sold throughout the simulation
    '''
    print("simulation started")
    total_revenue = 0
    sold_items = []
    for _ in range(m_const - 1):
        if heavy_items:        
                                                   # Phase 1 of the algorithm: offering heavy items one by one
            if random.uniform(0, 1) <= heavy_items[0].purchase_probability:
                heavy_items[0].inventory_level -= 1
                
                total_revenue += 1 / ( 1 - heavy_items[0].purchase_probability )
                if heavy_items[0].inventory_level <= 0:
                    sold_items.append(heavy_items.pop(0))
        else:                                                                       # Phase 2 of the algorithm: offering light items altogether
            if light_items:
                q0_approx = update_bundle_probabilities(light_items)
                probs = [o.purchase_probability for o in light_items]
                probs.append(q0_approx)
                for i in probs:
                    i = i / sum(probs)
                no_purchase_item = Item(0, 0, 0, q0_approx)
                light_items.append(no_purchase_item)
                item_bought = np.random.choice(light_items, 1,p=probs)[0]
                sold_items.append(item_bought)
                total_revenue += 1 / ( 1 - item_bought.purchase_probability ) 
                if item_bought == no_purchase_item:
                    light_items.remove(item_bought)
                else:
                    light_items[light_items.index(item_bought)].inventory_level -= 1
                    
                    if light_items[light_items.index(item_bought)].inventory_level == 0:
                        light_items.remove(item_bought)
                    light_items.remove(no_purchase_item)

    return total_revenue

def calculate_expected_revenue(bundle):
    if len(bundle) == 0: return 0
    q0_bundle = bisection(f, 0, 1, 500, bundle)
    print(q0_bundle)
    if len(bundle) == 1: 
        bundle[0].purchase_probability = solve_V( bisection(f, 0, 1, 500, [bundle[0]]) * math.e ** (bundle[0].quality - 1))
    
    else: 
        for item in bundle:
            item.purchase_probability = solve_V(q0_bundle * math.e ** ( 1 - item.quality))
    
    expected_revenue = 0
    for item in bundle:
        q_i = solve_V(q0_bundle * math.e ** (item.quality - 1))
        expected_revenue +=  q_i / ( 1 - q_i ) 
    return expected_revenue

def generate_item_power_set(iterable):

    return list(chain.from_iterable(combinations(iterable, r) for r in range(len(iterable) + 1)))

def construct_constrain_matrix(power_set_item, m_const, items):

    constrain_matrix = np.zeros((len(items), len(power_set_item)))
    for i in range(len(power_set_item)):
        q0_bundle = bisection(f, 0, 1, 500, power_set_item[i])

        for j in range(len(items)):
            purchase_probability = solve_V(q0_bundle * math.e ** (items[j].quality - 1))
            constrain_matrix[j][i] = purchase_probability

    result = constrain_matrix 
    for _ in range(m_const - 1):
        result = np.concatenate((result, constrain_matrix), axis = 1)    
           
    return result
def construct_probability_simplex_matrix(power_set_item, m_const):
    length = len(power_set_item)
    probability_simplex_matrix = np.zeros((m_const * length, m_const * length), dtype=np.uint8)
    counter = 0
    for row in range(0, m_const * length, length):
        for col in range(length):
            probability_simplex_matrix[row][(counter * length) + col] = 1

        counter += 1    

    return probability_simplex_matrix

def construct_objective_matrix(power_set_item, m_const):
    objective_vec = []
    for bundle in power_set_item:
        expected_bundle_revenue = calculate_expected_revenue(bundle)
        objective_vec.append(expected_bundle_revenue)
    objective_vec = np.array(objective_vec)
    result = objective_vec
    for _ in range(m_const - 1):
        result = np.append(result, objective_vec) 
    return result

def find_optimal_val(items, m_const):
    power_set_item = generate_item_power_set(items)
    inventory_level_vec = np.array([o.inventory_level for o in items])
    constrain_matrix = construct_constrain_matrix(power_set_item, m_const, items)
    probability_simplex_matrix = construct_probability_simplex_matrix(power_set_item, m_const)
    simplex_constraint = np.zeros(m_const * 2 ** len(items))
    for i in range(0, m_const * 2 ** len(items), 2 ** len(items)):
        simplex_constraint[i] = 1

    # combined_matrix = np.concatenate((constrain_matrix, probability_simplex_matrix), axis = 0)
    # combined_constraint = np.append(inventory_level_vec, simplex_constraint)
    objective_function_vec = construct_objective_matrix(power_set_item, m_const)
    sol = linprog(-objective_function_vec, A_ub=constrain_matrix, b_ub=inventory_level_vec, A_eq = probability_simplex_matrix, b_eq = simplex_constraint, bounds = (0,1))
    
    
    return sol

def main():
    items = []
    for _ in range(5):
        items.append(Item(random.randint(1, 50), random.randint(1, 50), random.uniform(-10 , 10), 0))
    
    r = []
    for m in range(10, 50, 1):
        revenue = 0
        for _ in range(100):
            heavy_items, light_items = assign_purchase_probabilities(items, 0.5)
            revenue += simulate(heavy_items, light_items, m)
        revenue /= 100
        r.append(revenue / -find_optimal_val(items, m).fun) 

    plt.plot(np.linspace(10,50,40), r, 'o', color='black')
    plt.title("m vs Comp Ratio lambda = 1/2")
    plt.xlabel("m (number of buyers)")
    plt.ylabel("Comp Ratio")
    plt.savefig('m vs Comp Ration alg1 s.png')

    r = []
    for inv_lvl in range(7, 16):
        items[0].inventory_level = inv_lvl
        revenue = 0
        for _ in range(100):
            heavy_items, light_items = assign_purchase_probabilities(items, 0.5)
            revenue += simulate(heavy_items, light_items, 25)
        revenue /= 100
        r.append(revenue / -find_optimal_val(items, 25).fun) 
    plt.plot(np.linspace(7,16,10), r, 'o', color='black')
    plt.title("inventory_level vs revenue, m = 25, lambda = 1/2")
    plt.xlabel("inventory level")
    plt.ylabel("comp ratio")
    plt.savefig('inventory level vs comp ratio alg1 s.png')

    r = []
    arr = np.linspace(0.3,0.9,60)
    for l in arr:
        revenue = 0
        for _ in range(100):
            heavy_items, light_items = assign_purchase_probabilities(items, l)
            revenue += simulate(heavy_items, light_items, 25)
        revenue /= 100
        r.append(revenue / -find_optimal_val(items, 25).fun)  
    plt.plot(np.linspace(0.3,0.9,60), r, 'o', color='black')
    plt.title("lambda vs Revenue, m = 25")
    plt.xlabel("lambda")
    plt.ylabel("Revenue")
    plt.savefig('lambda vs Revenue alg1 s.png')
# ---------------------------------------------------------------------------------------------------
    items = []
    for _ in range(5):
        inventory_lvl = random.randint(1, 50)
        items.append(Item(inventory_lvl, inventory_lvl, random.uniform(-10 , 10), 0))
    
    comp_ratio = []
    for m in range(10, 50):
        revenue = 0
        for _ in range(100):
            revenue += simulate_2(items, m, 0.5)
        revenue /= 100
        comp_ratio.append(revenue / -find_optimal_val(items,m).fun) 

    plt.plot(np.linspace(10,50,40), comp_ratio, 'o', color='black')
    plt.title("m vs comp ratio, lambda = 1/2, relative_weight")
    plt.xlabel("m (number of buyers)")
    plt.ylabel("Comp Ratio")
    plt.savefig('m vs comp ratio Alg2.png')

    comp_ratio = []
    for inv_lvl in range(7, 16):
        items[0].inventory_level = inv_lvl
        revenue = 0
        for _ in range(100):
            revenue += simulate_2(items, 25, 0.5)
        revenue /= 100
        comp_ratio.append(revenue / -find_optimal_val(items,25).fun) 
    plt.plot(np.linspace(15,30,15), comp_ratio, 'o', color='black')
    plt.title("inventory_level vs comp ratio, m = 25, lambda = 1/2, relative_weight")
    plt.xlabel("inventory level")
    plt.ylabel("comp ratio")
    plt.savefig('inventory level vs Revenue alg2 s .png')

    comp_ratio = []
    arr = np.linspace(0.3,0.9,60)
    for l in arr:
        revenue = 0
        for _ in range(100):

            revenue += simulate_2(items, 50, l)
        revenue /= 100
        comp_ratio.append(revenue / -find_optimal_val(items,25).fun)  
    plt.plot(np.linspace(0.3,0.9,60), comp_ratio, 'o', color='black')
    plt.title("lambda vs Revenue, m = 25")
    plt.xlabel("lambda")
    plt.ylabel("comp ratio")
    plt.savefig('lambda vs comp ratio alg2 s.png')

if __name__ == "__main__":
    main()    