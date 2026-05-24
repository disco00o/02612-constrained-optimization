import os
import numpy as np
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
from scipy.optimize import root
from matplotlib.lines import Line2D

FIGURE_DIR = "Exercise4-NLP/figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

# gradient and hessian
def grad_f(x, y): 
    dfdx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    dfdy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return np.array([dfdx, dfdy])

def hessian_f(x, y):
    d2fdx2 = 12*x**2 + 4*y - 42
    d2fdy2 = 4*x + 12*y**2 - 26
    d2fdxdy = 4*x + 4*y
    return np.array([
        [d2fdx2, d2fdxdy],
        [d2fdxdy, d2fdy2]])

# constraints and constraint gradients
def c1(x, y): 
    return (x + 2)**2 - y

def grad_c1(x, y): 
    return np.array([2*(x + 2), -1])

def hessian_c1(x, y):
    return np.array([
        [2, 0],
        [0, 0]
    ])

def c2(x, y): 
    return -4*x + 10*y

def grad_c2(x, y): 
    return np.array([-4, 10])

# feasibilty check for a given point
def is_feasible(x, y): 
    return c1(x, y) >= -1e-5 and c2(x, y) >= -1e-5

# define the three cases we need to solve
def sys_interior(vars):
    x, y = vars[0], vars[1]
    return grad_f(x, y)

def sys_boundary_c1(vars):
    x, y, lam = vars[0], vars[1], vars[2]
    eqs = grad_f(x, y) - lam * grad_c1(x, y)
    return [eqs[0], eqs[1], c1(x, y)]

def sys_boundary_c2(vars):
    x, y, lam = vars[0], vars[1], vars[2]
    eqs = grad_f(x, y) - lam * grad_c2(x, y)
    return [eqs[0], eqs[1], c2(x, y)]


def classify_stationary_point(x, y, loc):
    
    H = hessian_f(x, y)
    
    loc_lower = loc.lower()
    
    # interior points
    if "interior" in loc_lower:
        evals = eigvals(H)
        
        if np.all(evals > 0):
            return "local minimum"
        elif np.all(evals < 0):
            return "local maximum"
        else:
            return "saddle point"

    # boundary points
    else:
        gf = grad_f(x, y)
        
        if "parabola" in loc_lower:
            grad_c = np.array([2*(x + 2), -1])
            H_c = hessian_c1(x, y)
            tangent = np.array([1, 2*(x + 2)])
            
        elif "line" in loc_lower:
            grad_c = np.array([-4, 10])
            H_c = np.zeros((2, 2)) 
            tangent = np.array([10, 4])
            
        # calculate the KKT Multiplier (lambda)
        lam = gf[1] / grad_c[1]
        
        # calculate the Hessian of the Lagrangian
        H_L = H - lam * H_c
        
        # project the Hessian onto the tangent line
        projected_curvature = tangent.T @ H_L @ tangent
        
        if projected_curvature > 0:
            return "local minimum"
        elif projected_curvature < 0:
            return "local maximum"
        else:
            return "saddle point"

# define the grid of initial guesses
grid_x = np.linspace(-5, 5, 15)
grid_y = np.linspace(-5, 5, 15)

raw_pts = []

for x0 in grid_x:
    for y0 in grid_y:
        
        # interior case
        s1 = root(sys_interior, [x0, y0])
        if s1.success and is_feasible(s1.x[0], s1.x[1]): 
            raw_pts.append((s1.x[0], s1.x[1], "Interior"))
            
        # parabola boundary case
        s2 = root(sys_boundary_c1, [x0, y0, 0])
        if s2.success:
            x, y, lam = s2.x[0], s2.x[1], s2.x[2]
            if lam >= -1e-5 and is_feasible(x, y): 
                raw_pts.append((x, y, "Boundary (parabola)"))

        # line boundary case
        s3 = root(sys_boundary_c2, [x0, y0, 0])
        if s3.success:
            x, y, lam = s3.x[0], s3.x[1], s3.x[2]
            if lam >= -1e-5 and is_feasible(x, y): 
                raw_pts.append((x, y, "Boundary (line)"))

# remove duplicate points
unique_pts = set()

for pt in raw_pts:
    x, y, loc = pt
    # round to 3 decimal places to match nearly identical floats
    unique_pts.add((round(x, 3), round(y, 3), loc))

unique_pts = list(unique_pts)

points = []

for x, y, loc in unique_pts:
    classification = classify_stationary_point(x, y, loc)
    points.append((x,y,classification))


#### contour plot ####
x1 = np.linspace(-6, 6, 400)
x2 = np.linspace(-6, 6, 400)
X1, X2 = np.meshgrid(x1, x2)

# Himmelblau's Objective Function
Z = (X1**2 + X2 - 11)**2 + (X1 + X2**2 - 7)**2

# Constraints (Formatted so >= 0 is feasible, < 0 is infeasible)
C1 = (X1 + 2)**2 - X2
C2 = -4 * X1 + 10 * X2

# Create the Plot

plt.figure(figsize=(8, 6))

# Objective Function Contours
contour = plt.contour(X1, X2, Z, levels=50, alpha=0.6)
plt.clabel(contour, inline=True, fontsize=8)

# Shade Infeasible Regions (Where C1 < 0 or C2 < 0)
plt.contourf(X1, X2, C1, levels=[-np.inf, 0], colors='gray', alpha=0.3)
plt.contourf(X1, X2, C2, levels=[-np.inf, 0], colors='gray', alpha=0.3)

# Draw Dashed Boundary Lines
plt.contour(X1, X2, C1, levels=[0], colors='black', linestyles='dashed', linewidths=1.5)
plt.contour(X1, X2, C2, levels=[0], colors='black', linestyles='dashed', linewidths=1.5)

# Scatter Plot the Points
colors = {
    "local minimum": "tab:blue",
    "local maximum": "tab:red",
    "saddle point": "tab:green"
}

markers = {
    "local minimum": "o",
    "local maximum": "s",
    "saddle point": "x"
}

for p in points:
    point_type = p[2]
    plt.scatter(
        p[0],
        p[1],
        marker=markers[point_type],
        color=colors[point_type],
        s=100,
        label=point_type,
        zorder=5
    )

# Formatting and Legends
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.grid(True)

# Remove duplicate legend labels for points
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))

# Add custom constraint handles to the legend
constraint_handles = [
    Line2D([0], [0], color='black', linestyle='dashed', lw=1.5, label=r'$(x_1+2)^2 - x_2 \geq 0$'),
    Line2D([0], [0], color='black', linestyle='dashed', lw=1.5, label=r'$-4x_1 + 10x_2 \geq 0$')
]

all_handles = list(unique.values()) + constraint_handles
all_labels = list(unique.keys()) + [h.get_label() for h in constraint_handles]

plt.legend(all_handles, all_labels, loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(
        f"{FIGURE_DIR}/himmelblau_contour_stationary_points.png",
        dpi=300,
        bbox_inches="tight"
    )
plt.show()