import unittest
from src.constrained_min import *
from tests.examples import *
from src.utils import plot_path
from src.utils import plot_func_values
import matplotlib.pyplot as plt

class TestMinMethods(unittest.TestCase):
    def test_qp(self):
        func = qp
        x0 = np.array([0.1, 0.2, 0.7], dtype=np.float64)
        eq_constraints_mat = np.array([1, 1, 1], dtype=np.float64).reshape((1, 3))
        ineq_constraints = [qp_constraint_1, qp_constraint_2, qp_constraint_3]
        
        x_history, f_history = interior_pt(func=func, ineq_constraints=ineq_constraints, eq_constraints_mat=eq_constraints_mat, eq_constraints_rhs=None, x0=x0)
        print(f"\n Final x: {x_history[-1]} \n Value f(x): {f_history[-1]}")
        print(f"\n Constraints validation:")
        qp_constraints_validation(x_history[-1])
        plot_func_values(f_history)
        plot_path(x_history, 1)

    def test_lp(self):
        x0 = np.array([0.5, 0.75], dtype=np.float64)
        func = lp
        barrier_func = f2_phi
        eq_constraints_mat = None
        eq_constraints_rhs = None 
        ineq_constraints = [lp_constraint_1, lp_constraint_2, lp_constraint_3, lp_constraint_4]

        x_history, f_history = interior_pt(func=func, ineq_constraints=ineq_constraints, eq_constraints_mat=eq_constraints_mat, eq_constraints_rhs=eq_constraints_rhs, x0=x0)
        print(f"\n Final x: {x_history[-1]} \n Value f(x): {f_history[-1]}")
        print(f"\n Constraints validation:")
        lp_constraints_validation(x_history[-1])
        plot_func_values(f_history)
        plot_path(x_history,2)

if __name__ == '__main__':
    unittest.main()
