from dataclasses import dataclass
import numpy as np
from sympy import lambdify
from scipy.optimize import fsolve
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

    def __eq__(self, o: object) -> bool:

        return ( self.unit_price == o.unit_price and 
               self.inventory_level == o.inventory_level and 
               self.quantity == o.quantity and 
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
        item.purchase_probability = 1 - bisection(f, 0, 1, 10000, [item])
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
    func = lambda y : y * math.exp(y / (1-y)) - x
    y_initial_guess = 0.5
    return fsolve(func, y_initial_guess)


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
    q0_approx = bisection(f, 0, 1, 10000, light_items)
    for item in light_items:
        item.purchase_probability = solve_V(q0_approx * np.e ** (item.quality - 1))
    return q0_approx

def simulate(heavy_items, light_items, m_const):
    '''
    
    '''
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
                no_purchase_item = Item(0, 0, 0, q0_approx)
                light_items.append(no_purchase_item)
                item_bought = np.random.choice(light_items, probs)
                sold_items.append(item_bought)

                if item_bought == no_purchase_item:
                    light_items.remove(item_bought)
                else:
                    light_items.remove(item_bought)
                    light_items.remove(no_purchase_item)

