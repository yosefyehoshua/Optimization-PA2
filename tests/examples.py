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

def f1_phi(x0):
    x, y, z = x0
    f = - (np.log(x) + np.log(y) + np.log(z))
    g = np.array([- 1 / x, - 1 / y, - 1 / z]).reshape((3,1))
    # h = np.array([[1 / x ** 2, 0, 0], [0, 1 / y ** 2, 0], [0, 0, 1 / z ** 2]])
    h = np.zeros((3,3))
    for i in range(3):
        h[i,i] = (g[i,0]/(-x0[i])) ** 2

    return f, g, h


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




def f2_phi(x0):
    x, y = x0
    f = - (np.log(x+y-1) + np.log(1-y) + np.log(2-x) + np.log(y))

    g_1 = np.array([-1/(x+y-1), -1/(x+y-1)]).reshape((2,1))
    g_2 = np.array([0, 1/(1-y)]).reshape((2,1))
    g_3 = np.array([-1/(x-2), 0]).reshape((2,1))
    g_4 = np.array([0, -1/y]).reshape((2,1))
    g = g_1 + g_2 + g_3 + g_4

    h_1 = np.matmul(g_1, g_1.T) / (-x-y+1)**2
    h_2 = np.matmul(g_2, g_2.T) / (-1+y)**2
    h_3 = np.matmul(g_3, g_3.T) / (-2+x)**2
    h_4 = np.matmul(g_4, g_4.T) / (-y)**2
    h = h_1 + h_2 + h_3 + h_4

    return f, g, h


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




# # Quadratic function definition
# def qp_function(x):
#     f = x[0] ** 2  + x[1] ** 2 + (x[2] + 1) ** 2
#     g = np.array([2 * x[0], 2 * x[1], 2 * x[2] + 2]).transpose()
#     h = np.array([
#         [2, 0, 0],
#         [0, 2, 0],
#         [0, 0, 2]
#     ])
#     return f, g, h


# # Quadratic function inequalties definition
# def qp_ineq1(x):
#     f = -x[0]
#     g = np.array([-1, 0, 0]).transpose()
#     h = np.array([
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]
#     ])
#     return f, g, h


# def qp_ineq2(x):
#     f = -x[1]
#     g = np.array([0, -1, 0]).transpose()
#     h = np.array([
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]
#     ])
#     return f, g, h


# def qp_ineq3(x):
#     f = -x[2]
#     g = np.array([0, 0, -1]).transpose()
#     h = np.array([
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]
#     ])
#     return f, g, h


# # Linear function definition
# def lp_function(x):
#     f = -x[0] - x[1]
#     g = np.array([-1, -1]).transpose()
#     h = np.array([
#         [0,0],
#         [0,0]
#     ])
#     return f, g, h


# # Linear function inequalties definition
# def lp_ineq1(x):
#     f = -x[0] - x[1] + 1
#     g = np.array([-1, -1]).transpose()
#     h = np.array([
#         [0, 0], 
#         [0, 0]
#     ])
#     return f, g, h


# def lp_ineq2(x):
#     f = x[1] - 1
#     g = np.array([0, 1]).transpose()
#     h = np.array([
#         [0, 0], 
#         [0, 0]
#     ])
#     return f, g, h


# def lp_ineq3(x):
#     f = x[0] - 2
#     g = np.array([1, 0]).transpose()
#     h = np.array([
#         [0, 0],
#         [0, 0]
#     ])
#     return f, g, h
 
 
# def lp_ineq4(x):
#     f = -x[1]
#     g = np.array([0, -1]).transpose()
#     h = np.array([
#         [0, 0],
#         [0, 0]
#     ])
#     return f, g, h