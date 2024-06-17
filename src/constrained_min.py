# src/constrained_min.py
import numpy as np

class Constraint:
    def __init__(self, func, grad, hess):
        self.func = func
        self.grad = grad
        self.hess = hess

def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, tol=1e-8, max_iter=100):
    def barrier(ineq_constraints, x):
        return -np.sum([np.log(-g.func(x)) for g in ineq_constraints])

    def grad_barrier(ineq_constraints, x):
        return -np.sum([g.grad(x) / g.func(x) for g in ineq_constraints], axis=0)

    def hess_barrier(ineq_constraints, x):
        return np.sum([np.outer(g.grad(x), g.grad(x)) / g.func(x)**2 - g.hess(x) / g.func(x) for g in ineq_constraints], axis=0)

    def newton_step(t, func, x):
        grad = t * func.grad(x) + grad_barrier(ineq_constraints, x)
        hess = t * func.hess(x) + hess_barrier(ineq_constraints, x)
        return np.linalg.solve(hess, -grad)

    m = len(ineq_constraints)
    t = 1.0
    mu = 10.0
    x = x0

    for _ in range(max_iter):
        for _ in range(100):
            step = newton_step(t, func, x)
            alpha = 1.0
            while any(g.func(x + alpha * step) >= 0 for g in ineq_constraints):
                alpha *= 0.5
            x += alpha * step
            if np.linalg.norm(step) < tol:
                break

        t *= mu
        if m / t < tol:
            break

    return x
