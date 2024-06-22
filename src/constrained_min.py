import numpy as np

def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, max_iter=10):
    t = 1.0
    x = x0.copy()
    f_x, g_x, h_x = update_barrier_values(func, ineq_constraints, x0, t)
    
    x_history, f_history = [x0], [func(x0)[0]]
    outer_iter = len(ineq_constraints) / t
    
    while outer_iter > 1e-8:
        for _ in range(max_iter):
            p = solve_kkt(eq_constraints_mat, g_x, h_x)
            alpha = wolfe_update(func, x, f_x, g_x, p, c1=0.01, rho=0.5)
            x = x + alpha * p
            f_x_next, g_x_next, h_x_next = update_barrier_values(func, ineq_constraints, x, t)
            if 0.5 * np.linalg.norm(g_x_next, 2) ** 2 < 1e-8: break
            
            f_x = f_x_next
            g_x = g_x_next
            h_x = h_x_next
            
        x_history.append(x)
        f_history.append(func(x)[0])
        t *= 10
        outer_iter = len(ineq_constraints) / t
    return x_history, f_history


def update_barrier_values(func, ineq_constraints, x0, t):
        f_x, g_x, h_x = func(x0)
        f_x_phi, g_x_phi, h_x_phi = log_barrier(ineq_constraints, x0)
        return t * f_x + f_x_phi, t * g_x + g_x_phi, t * h_x + h_x_phi
    
def log_barrier(ineq_constraints, x0):
    dim = x0.size
    log_sum = 0
    grad_sum = np.zeros(dim)
    hess_sum = np.zeros((dim, dim))

    for func in ineq_constraints:
        val, grad, hess = func(x0)
        inv_val = -1 / val
        log_sum += np.log(-val)
        grad_sum += inv_val * grad

        grad_div_val = grad / val
        grad_outer_prod = np.outer(grad_div_val, grad_div_val)
        hess_sum += (hess * val - grad_outer_prod) / (val ** 2)
    
    return -log_sum, grad_sum, -hess_sum

def solve_kkt(eq_constraints_mat, g_x, h_x):
    if eq_constraints_mat is not None:
        n_constraints =  eq_constraints_mat.shape[0]
        A = eq_constraints_mat
        kkt = np.block([[h_x, A.T],[A, np.zeros((n_constraints, n_constraints))]])
        B = np.vstack([np.hstack([h_x, A.T]), np.hstack([A, np.zeros((1, 1))])])
        rhs = np.concatenate([-g_x, np.zeros(n_constraints)])
        return np.linalg.solve(kkt, rhs)[:A.shape[1]]
    else:
        B = h_x
        rhs = -g_x
        return np.linalg.solve(B, rhs)

def wolfe_update(func, x, f_x, gradient, p, c1, rho):
    alpha = 1.0
    while func(x + alpha * p)[0] > f_x + c1 * alpha * np.dot(gradient.T, p):
        alpha *= rho
        if alpha < 1e-3: break
    return alpha

