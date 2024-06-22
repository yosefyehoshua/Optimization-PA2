import numpy as np

def qp(x0):
    x, y, z = x0
    f = x**2 + y**2 + (z+1)**2
    g = np.array([2*x, 2*y, 2*z+2]).T
    h = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    return f, g, h

def qp_constraint_1(x0):
    x, _, _ = x0
    f = -x
    g = np.array([-1, 0, 0]).T
    h = np.zeros((3,3))
    return f, g, h

def qp_constraint_2(x0):
    _, y, _ = x0
    f = -y
    g = np.array([0, -1, 0]).T
    h = np.zeros((3,3))
    return f, g, h

def qp_constraint_3(x0):
    _, _, z = x0
    f = -z
    g = np.array([0, 0, -1]).T
    h = np.zeros((3,3))
    return f, g, h

def qp_constraints(x0):
    qp_constraint = lambda i: (lambda x0: (-x0[i], np.eye(3)[i] * -1, np.zeros((3, 3))))
    return qp_constraint

def lp(x0):
    x, y = x0
    f = - (x + y)
    g = np.array([-1, -1]).T
    h = np.zeros((2,2))
    return f, g, h

def lp_constraint_1(x0):
    x, y = x0
    f = - (x + y - 1)
    g = np.array([-1, -1]).T
    h = np.zeros((2,2))
    return f, g, h

def lp_constraint_2(x0):
    _, y = x0
    f = y - 1
    g = np.array([0, 1]).T
    h = np.zeros((2,2))
    return f, g, h

def lp_constraint_3(x0):
    x, _ = x0
    f = x - 2
    g = np.array([1, 0]).T
    h = np.zeros((2,2))
    return f, g, h

def lp_constraint_4(x0):
    _, y = x0
    f = -y
    g = np.array([0, -1]).T
    h = np.zeros((2,2))
    return f, g, h

def lp_constraints(x0):
    lp_constraint = lambda i: (lambda x0: (
    -(x0[0] + x0[1] - 1) 
    if i == 0 else (x0[1] - 1) 
    if i == 1 else (x0[0] - 2) 
    if i == 2 else -x0[1], 
    
    np.array([-1, -1]).T 
    if i == 0 else np.array([0, 1]).T 
    if i == 1 else np.array([1, 0]).T 
    if i == 2 else np.array([0, -1]).T,
    np.zeros((2, 2))
    ))



def qp_constraints_validation(x_final):
    x, y, z = x_final
    c1 = np.isclose(x+y+z, 1)
    c2 = (x >= 0)
    c3 = (y >= 0)
    c4 = (z >= 0)
    print(f"Constraint 1: x + y + z = 1 {c1}")
    print(f"Constraint 2: x >= 0 {c2}")
    print(f"Constraint 3: y >= 0 {c3}")
    print(f"Constraint 4: z >= 0 {c4}")



def lp_constraints_validation(x_final):
    x, y = x_final
    c1 = (y >= -x+1)
    c2 = (y <= 1)
    c3 = (x <= 2)
    c4 = (y >= 0)
    print(f"Constraint 1: y >= -x + 1 {c1}")
    print(f"Constraint 2: y <= 1 {c2}")
    print(f"Constraint 3: x <= 2 {c3}")
    print(f"Constraint 4: y >= 0 {c4}")



