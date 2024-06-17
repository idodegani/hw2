import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from src.constrained_min import interior_pt, Constraint

# Define the objective function and constraints for QP
def qp_func(x):
    return x[0]**2 + x[1]**2 + (x[2] + 1)**2

def qp_constraints():
    return [
        Constraint(lambda x: -x[0], lambda x: np.array([-1, 0, 0]), lambda x: np.zeros((3, 3))),
        Constraint(lambda x: -x[1], lambda x: np.array([0, -1, 0]), lambda x: np.zeros((3, 3))),
        Constraint(lambda x: -x[2], lambda x: np.array([0, 0, -1]), lambda x: np.zeros((3, 3)))
    ]

# Define the objective function and constraints for LP
def lp_func(x):
    return -(x[0] + x[1])

def lp_constraints():
    return [
        Constraint(lambda x: x[1] + x[0] - 1, lambda x: np.array([1, 1]), lambda x: np.zeros((2, 2))),
        Constraint(lambda x: -x[1] + 1, lambda x: np.array([0, -1]), lambda x: np.zeros((2, 2))),
        Constraint(lambda x: -x[0] + 2, lambda x: np.array([-1, 0]), lambda x: np.zeros((2, 2))),
        Constraint(lambda x: -x[1], lambda x: np.array([0, -1]), lambda x: np.zeros((2, 2)))
    ]

# Optimization function with tracking
def optimize_with_tracking(func, x0, constraints, title, is_qp):
    path = []
    constraints_dict = [{'type': 'ineq', 'fun': con.func, 'jac': con.grad, 'hess': con.hess} for con in constraints]
    
    def record(x):
        path.append(x.copy())
    
    options = {'maxiter': 100, 'disp': True}
    result = minimize(func, x0, method='SLSQP', constraints=constraints_dict, callback=record, options=options)

    # Plot results
    fig = plt.figure(figsize=(12, 10))
    if is_qp:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*np.array(path).T, color='r', marker='o', label='Path')
        ax.scatter(result.x[0], result.x[1], result.x[2], color='b', s=100, label='Final Candidate')
        ax.set_title(f'{title} Results')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        ax = fig.add_subplot(111)
        ax.plot(*np.array(path).T, 'ro-', label='Path')
        ax.scatter(result.x[0], result.x[1], color='b', s=100, label='Final Candidate')
        ax.set_title(f'{title} Results')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    ax.legend()
    plt.show()
    return result

def main():
    # Optimize and plot results for QP
    x0_qp = np.array([0.1, 0.2, 0.7])
    qp_result = optimize_with_tracking(qp_func, x0_qp, qp_constraints(), 'Quadratic Programming', is_qp=True)
    print("QP Final Candidate:", qp_result.x)

    # Optimize and plot results for LP
    x0_lp = np.array([0.5, 0.75])
    lp_result = optimize_with_tracking(lp_func, x0_lp, lp_constraints(), 'Linear Programming', is_qp=False)
    print("LP Final Candidate:", lp_result.x)

if __name__ == "__main__":
    main()
