from pathlib import Path
import json


import matplotlib.pyplot as plt
import numpy as np


def main():
    json_path = Path("poisson_grid_search.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    min_chamfer = 1e10
    min_datum = None
    for datum in data:
        if datum["chamfer"] < min_chamfer:
            min_chamfer = datum["chamfer"]
            min_datum = datum
    print(f"min_chamfer: {min_chamfer}")
    print(f"min_datum: {min_datum}")

    # Fill confusion matrix.
    col_vals = [8, 9, 10, 11, 12]
    row_vals = [0.4, 0.3, 0.2]
    conf_matrix = np.zeros((len(row_vals), len(col_vals)))
    for datum in data:
        min_density = datum["poisson_min_density"]
        poisson_depth = datum["poisson_depth"]
        if min_density not in row_vals or poisson_depth not in col_vals:
            continue
        row_idx = row_vals.index(min_density)
        col_idx = col_vals.index(poisson_depth)
        conf_matrix[row_idx, col_idx] = datum["chamfer"]

    # Print the confusion matrix using Matplotlib
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=f"{conf_matrix[i, j]:.2f}",
                va="center",
                ha="center",
                size="xx-large",
            )
    ax.set_xticklabels([""] + [str(v) for v in col_vals])
    ax.set_yticklabels([""] + [str(v) for v in row_vals])

    plt.xlabel("Poisson Depth", fontsize=18)
    plt.ylabel("Min Density", fontsize=18)
    plt.show()


if __name__ == "__main__":
    main()
