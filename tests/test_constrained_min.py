# tests/test_constrained_min.py
import unittest
import numpy as np
from src.constrained_min import interior_pt, Constraint
class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        func = lambda x: x[0]**2 + x[1]**2 + (x[2] + 1)**2
        func.grad = lambda x: np.array([2*x[0], 2*x[1], 2*(x[2] + 1)])
        func.hess = lambda x: np.diag([2, 2, 2])

        ineq_constraints = [
            Constraint(lambda x: -x[0], lambda x: np.array([-1, 0, 0]), lambda x: np.zeros((3, 3))),
            Constraint(lambda x: -x[1], lambda x: np.array([0, -1, 0]), lambda x: np.zeros((3, 3))),
            Constraint(lambda x: -x[2], lambda x: np.array([0, 0, -1]), lambda x: np.zeros((3, 3)))
        ]

        eq_constraints_mat = np.array([[1, 1, 1]])
        eq_constraints_rhs = np.array([1])
        x0 = np.array([0.1, 0.2, 0.7])

        result = interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)
        self.assertTrue(np.allclose(result.sum(), 1))

    def test_lp(self):
        func = lambda x: -(x[0] + x[1])
        func.grad = lambda x: np.array([-1, -1])
        func.hess = lambda x: np.zeros((2, 2))

        ineq_constraints = [
            Constraint(lambda x: x[1] + x[0] - 1, lambda x: np.array([1, 1]), lambda x: np.zeros((2, 2))),
            Constraint(lambda x: -x[1] + 1, lambda x: np.array([0, -1]), lambda x: np.zeros((2, 2))),
            Constraint(lambda x: -x[0] + 2, lambda x: np.array([-1, 0]), lambda x: np.zeros((2, 2))),
            Constraint(lambda x: -x[1], lambda x: np.array([0, -1]), lambda x: np.zeros((2, 2)))
        ]

        eq_constraints_mat = np.empty((0, 2))
        eq_constraints_rhs = np.empty(0)
        x0 = np.array([0.5, 0.75])

        result = interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)
        self.assertTrue(np.allclose(result, [2, 1]))

if __name__ == "__main__":
    unittest.main()
