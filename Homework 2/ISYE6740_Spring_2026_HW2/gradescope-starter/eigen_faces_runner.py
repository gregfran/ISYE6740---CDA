import numpy as np
import matplotlib.pyplot as plt
import os
from eigen_faces import EigenFaces

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # algo class call
    ef_algo = EigenFaces()
    result = ef_algo.run()

    # save projection residuals to txt
    with open("results/projection_residuals.txt", 'w') as f:
        f.write(f"s11: {round(result.s11, 0)}\n")
        f.write(f"s12: {round(result.s12, 0)}\n")
        f.write(f"s21: {round(result.s21, 0)}\n")
        f.write(f"s22: {round(result.s22, 0)}\n")

    # plot results
    fig, ax = plt.subplots(2, 6, figsize=(18, 12))
    for i in range(6):
        ax[0, i].imshow(result.subject_1_eigen_faces[i], cmap='gray')
        ax[0, i].set_title(f"Subj 1 - Eigenface {i+1}")
        ax[1, i].imshow(result.subject_2_eigen_faces[i], cmap='gray')
        ax[1, i].set_title(f"Subj 2 - Eigenface {i+1}")

    plt.tight_layout()
    plt.savefig("results/eigen_faces.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()