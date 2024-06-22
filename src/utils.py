import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tests.examples import *


def plot_func_values(f_history):
    f_history = np.array(f_history)
    plt.plot(f_history)
    plt.scatter(range(len(f_history)), f_history, color='blue', marker='o', s=50, label='intermediate values')
    plt.scatter(0, f_history[0], color='red', marker='o', s=50, label='Start value')
    plt.scatter(len(f_history) - 1, f_history[-1], color='gold', marker='o', s=50, label='Final value')

    plt.xlabel('Iteration')
    plt.ylabel('Function value')
    plt.title('Function value convergence')
    plt.legend()
    plt.show()

def plot_path(x_history, func_type):
    plot_functions = {
        1: plot_3d_minimization_path,
        2: plot_2d_minimization_path
    }
    plot_func = plot_functions.get(func_type)
    if plot_func:
        plot_func(x_history)

def plot_3d_minimization_path(x_history):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    draw_3d_feasible_region(ax)
    plot_minimization_path_3d(ax, x_history)
    
    plt.show()

def draw_3d_feasible_region(ax):
    vertices = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    poly_3d_collection = Poly3DCollection([vertices], alpha=0.25)
    ax.add_collection3d(poly_3d_collection)

def plot_2d_minimization_path(x_history):
    fig, ax = plt.subplots()
    draw_2d_feasible_region(ax)
    plot_minimization_path_2d(ax, x_history)
    plt.show()

def draw_2d_feasible_region(ax):
    x = np.linspace(-0.5, 2.5, 200)
    y = np.linspace(-0.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)
    
    constraints = [X + Y - 1,
                   Y - 1,
                   X - 2, 
                   Y]

    for constraint, color in zip(constraints, ['green', 'red', 'blue', 'orange']):
        ax.contour(X, Y, constraint, levels=[0], colors=color, linestyles='dashed', linewidths=1)

    x_fill = [0, 2, 2, 1]
    y_fill = [1, 1, 0, 0]
    ax.fill(x_fill, y_fill, facecolor='lightsteelblue', edgecolor='k', alpha=0.5)
    

def plot_minimization_path_3d(ax, x_history):
    x_history = np.array(x_history)
    ax.plot(x_history[:, 0], x_history[:, 1], x_history[:, 2], 
            color="k")
    ax.scatter(x_history[:, 0], x_history[:, 1], x_history[:, 2], 
            color="k", label='intermediate points')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Minimization Path')

    # mark the start point
    ax.scatter(x_history[0][0], x_history[0][1], x_history[0][2], 
               color='red', marker='o', s=50, label='Start point')
    # mark the final point
    ax.scatter(x_history[-1][0], x_history[-1][1], x_history[-1][2], 
               color='gold', marker='o', s=50, label='Final candidate')
    ax.legend()
    
def plot_minimization_path_2d(ax, x_history):
    x_history = np.array(x_history)
    ax.plot(x_history[:, 0], x_history[:, 1])
    ax.scatter(x_history[:, 0], x_history[:, 1], color='blue', marker='o', s=20, label='intermediate points')
    ax.scatter(x_history[0][0], x_history[0][1], color='red', marker='o', s=60, label='Start point')
    ax.scatter(x_history[-1][0], x_history[-1][1], color='gold', marker='o', s=60, label='Final candidate')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Minimization Path')
    ax.legend()


