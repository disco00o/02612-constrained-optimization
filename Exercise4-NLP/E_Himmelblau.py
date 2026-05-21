import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root


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
        levels=50
    )

    plt.clabel(
        contour,
        inline=True,
        fontsize=8
    )

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
            label=point_type
        )

    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    plt.title(
        "Himmelblau Function: Contours and Stationary Points"
    )

    plt.grid(True)

    # Remove duplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()

    unique = dict(zip(labels, handles))

    plt.legend(
        unique.values(),
        unique.keys()
    )

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_DIR}/himmelblau_contour_stationary_points.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    points = find_stationary_points()

    print("Stationary points:\n")

    for p in points:

        print(
            f"x = {p}, "
            f"f(x) = {himmelblau(p):.6f}, "
            f"type = {classify_stationary_point(p)}"
        )

    with open(
        f"{FIGURE_DIR}/himmelblau_stationary_points.txt",
        "w"
    ) as f:

        f.write(
            "Stationary points for Himmelblau function\n\n"
        )

        for p in points:

            f.write(
                f"x = {p}, "
                f"f(x) = {himmelblau(p):.6f}, "
                f"type = {classify_stationary_point(p)}\n"
            )

    plot_himmelblau(points)