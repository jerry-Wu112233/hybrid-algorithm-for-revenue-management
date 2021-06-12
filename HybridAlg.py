from dataclasses import dataclass
import numpy as np
import sympy as sp
import random
import math

@dataclass
class Item:
    '''A data class with four attributes
    
    unit_price: a positive real number that stores the price of an item
    inventory_level: a positive integer that describes the current inventory level of an item
    quality: a real number that describes the quality of an item (could be negative or positive)
    purchase_probability: a real number in [0,1]
    '''
    unit_price: float 
    inventory_level: int 
    quality: float 
    purchase_probability: float 

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
        item.purchase_probability = 1 - solve_q0_single(item.quality)
        if (item.purchase_probability >= lambda_const):
            heavy_items.append(item)
        else:
            light_items.append(item)    
        
    return heavy_items, light_items
    
def solve_q0_single(theta):
    '''Computes solves the equation (4) within the paper to compute the no-purchase probability q0.

    Helper method for computing the no-purchase probability.

    Args:
        theta: the quality attribute of the single item in focus.

    Returns:
        The no-purchase probability associated with S where S := {i} where i is the item in focus.     
    '''
    q0,y = sp.symbols('q0, y')
    
    eq1 = sp.Eq(y, 1 - q0)
    eq2 = sp.Eq(y * np.e ** (y / (1 - y)),  q0 * np.e ** (theta - 1))
    soln = sp.solve((eq1,eq2), (q0, y))
    return soln[0][0].evalf()

def solve_V(x):
    y = sp.symbols('y')
    soln = sp.solve(sp.Eq(y * np.e ** (y / (1 - y)), x), y)
    return soln[0][0].evalf()

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
  
    a_n = a
    b_n = b
    for n in range(1,N+1):
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
    function_value = 0
    for item in light_items:
        function_value += solve_V(q0 * np.e ** (item.quality - 1))

    return function_value - 1 + q0

def update_bundle_probabilities(light_items):
    q0_approx = bisection(f, 0, 1, 10000, light_items)
    for item in light_items:
        item.purchase_probability = solve_V(q0_approx * np.e ** (item.quality - 1))
    return q0_approx

def simulate(heavy_items, light_items, m_const):
    sold_items = []
    for _ in m_const:
        if heavy_items:                                                             # Phase 1 of the algorithm: offering heavy items one by one
            if random.uniform(0, 1) <= heavy_items[0].purchase_probability:
                sold_items.append(heavy_items.pop())
              
        else:                                                                       # Phase 2 of the algorithm: offering light items altogether
            if light_items:
                q0_approx = update_bundle_probabilities(light_items)
                probs = (o.purchase_probability for o in light_items)
                probs.append(q0_approx)
                item_bought = np.random.choice(light_items, probs)
