import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from matplotlib.lines import Line2D


FIGURE_DIR = "Exercise4-NLP/figures"

os.makedirs(FIGURE_DIR, exist_ok=True)


def himmelblau(x):
    x1, x2 = x

    return (
        (x1**2 + x2 - 11)**2
        + (x1 + x2**2 - 7)**2
    )


def grad_himmelblau(x):

    x1, x2 = x

    df_dx1 = (
        4 * x1 * (x1**2 + x2 - 11)
        + 2 * (x1 + x2**2 - 7)
    )

    df_dx2 = (
        2 * (x1**2 + x2 - 11)
        + 4 * x2 * (x1 + x2**2 - 7)
    )

    return np.array([
        df_dx1,
        df_dx2
    ])


def hess_himmelblau(x):

    x1, x2 = x

    h11 = 12 * x1**2 + 4 * x2 - 42
    h22 = 4 * x1 + 12 * x2**2 - 26
    h12 = 4 * x1 + 4 * x2

    return np.array([
        [h11, h12],
        [h12, h22]
    ])


def classify_stationary_point(x):

    H = hess_himmelblau(x)

    eigvals = np.linalg.eigvals(H)

    tol = 1e-8

    if np.all(eigvals > tol):
        return "local minimum"

    elif np.all(eigvals < -tol):
        return "local maximum"

    else:
        return "saddle point"


def is_feasible(x, tol=1e-6):
    x1, x2 = x
    c1 = (x1 + 2)**2 - x2 >= -tol
    c2 = -4 * x1 + 10 * x2 >= -tol
    
    return c1 and c2


def find_stationary_points():

    guesses = []

    for x1 in np.linspace(-6, 6, 9):
        for x2 in np.linspace(-6, 6, 9):

            guesses.append(
                np.array([x1, x2])
            )

    points = []

    for guess in guesses:

        sol = root(
            grad_himmelblau,
            guess
        )

        if sol.success:

            x_star = sol.x

            # Only add the point if it falls within the feasible region
            if is_feasible(x_star):
                
                # Avoid duplicates
                if not any(
                    np.linalg.norm(x_star - p) < 1e-6
                    for p in points
                ):
                    points.append(x_star)

    return points


def plot_himmelblau(points):

    x1 = np.linspace(-6, 6, 400)
    x2 = np.linspace(-6, 6, 400)

    X1, X2 = np.meshgrid(x1, x2)

    Z = (
        (X1**2 + X2 - 11)**2
        + (X1 + X2**2 - 7)**2
    )

    plt.figure(figsize=(8, 6))

    contour = plt.contour(
        X1,
        X2,
        Z,
        levels=50,
        alpha=0.6
    )

    plt.clabel(
        contour,
        inline=True,
        fontsize=8
    )

    C1 = (X1 + 2)**2 - X2
    C2 = -4 * X1 + 10 * X2

    plt.contourf(X1, X2, C1, levels=[-np.inf, 0], colors='gray', alpha=0.3)
    plt.contourf(X1, X2, C2, levels=[-np.inf, 0], colors='gray', alpha=0.3)
    
    plt.contour(X1, X2, C1, levels=[0], colors='black', linestyles='dashed', linewidths=1.5)
    plt.contour(X1, X2, C2, levels=[0], colors='black', linestyles='dashed', linewidths=1.5)

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

        point_type = classify_stationary_point(p)

        plt.scatter(
            p[0],
            p[1],
            marker=markers[point_type],
            color=colors[point_type],
            s=100,
            label=point_type,
            zorder=5
        )

    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    plt.grid(True)

    # Remove duplicate legend labels for points and add constraints
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    
    constraint_handles = [
        Line2D([0], [0], color='black', linestyle='dashed', lw=1.5, label=r'$(x_1+2)^2 - x_2 \geq 0$'),
        Line2D([0], [0], color='black', linestyle='dashed', lw=1.5, label=r'$-4x_1 + 10x_2 \geq 0$')
    ]
    
    all_handles = list(unique.values()) + constraint_handles
    all_labels = list(unique.keys()) + [h.get_label() for h in constraint_handles]

    plt.legend(
        all_handles,
        all_labels,
        loc='upper right',
        fontsize=9
    )

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_DIR}/himmelblau_constrained_stationary_points.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    points = find_stationary_points()

    print("Feasible Stationary points:\n")

    for p in points:

        print(
            f"x = {p}, "
            f"f(x) = {himmelblau(p):.6f}, "
            f"type = {classify_stationary_point(p)}"
        )

    with open(
        f"{FIGURE_DIR}/himmelblau_constrained_stationary_points.txt",
        "w"
    ) as f:

        f.write(
            "Feasible stationary points for Himmelblau function\n\n"
        )

        for p in points:

            f.write(
                f"x = {p}, "
                f"f(x) = {himmelblau(p):.6f}, "
                f"type = {classify_stationary_point(p)}\n"
            )

    plot_himmelblau(points)