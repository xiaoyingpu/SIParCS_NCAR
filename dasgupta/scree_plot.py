import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import manifold, datasets

# von http://stats.stackexchange.com/questions/12819/how-to-draw-a-scree-plot-in-python

matrix_f = "dm_psl.txt"

with open(matrix_f) as f:
    A = np.loadtxt(f)

def do_scree_plot(A):
    num_vars = 191
    U, S, V = np.linalg.svd(A)
    eigvals = S**2 / np.cumsum(S)[-1]

    fig = plt.figure(figsize=(8,5))
    sing_vals = np.arange(num_vars) + 1
    sing_vals = sing_vals[:9]
    eigvals = eigvals[:9]
    plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    leg.draggable(state=True)
    plt.show()


fold = manifold.MDS(n_components=2, dissimilarity='precomputed')
fold.fit_transform(A)
print fold.stress_
