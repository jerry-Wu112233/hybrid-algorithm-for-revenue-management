from dataclasses import dataclass
import numpy as np
import sympy as sp

@dataclass
class Item:
    unit_price: float 
    inventory_level: int 
    quality: float 
    purchase_probability: float 

def assign_purchase_probabilities(items):
    for item in items:
        item.purchase_probability = 1 - solve_q0(item.quality)
    items.sort(key = lamda x: x.quality, reverse = True)

def solve_q0(theta):
    q0,y = sp.symbols('q0, y')
    
    eq1 = sp.Eq(y, 1 - q0)
    eq2 = sp.Eq(y * np.e ** (y / (1 - y)),  q0 * np.e ** (theta - 1))
    soln = sp.solve((eq1,eq2), (q0, y))
    return soln[0][0].evalf()

def solve_V(x):
    y = sp.symbols('y')
    soln = sp.solve(sp.Eq(y * np.e ** (y / (1 - y)), x), y)
    return soln[0][0].evalf()

def simulate(items):
    pass        
        
