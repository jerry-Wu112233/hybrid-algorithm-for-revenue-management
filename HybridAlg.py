from dataclasses import dataclass
import numpy as np
import sympy as sp
import random

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
    q0,y = sp.symbols('q0, y')
    
    eq1 = sp.Eq(y, 1 - q0)
    eq2 = sp.Eq(y * np.e ** (y / (1 - y)),  q0 * np.e ** (theta - 1))
    soln = sp.solve((eq1,eq2), (q0, y))
    return soln[0][0].evalf()

def solve_V(x):
    y = sp.symbols('y')
    soln = sp.solve(sp.Eq(y * np.e ** (y / (1 - y)), x), y)
    return soln[0][0].evalf()

def solve_bundle_probabilities(light_items):

    pass

def simulate(heavy_items, light_items, m_const):
    sold_items = []
    for _ in m_const:
        if heavy_items:
            if random.uniform(0, 1) <= heavy_items[0].purchase_probability:
                sold_items.append(heavy_items.pop())
            else:
                continue    
        else:
            if light_items:
                continue
                pass
            
        
