from sklearn.decomposition import PCA as skpca
import numpy as np
from scipy.io import loadmat
import os
from scipy.spatial.distance import cdist # to easily calc euc dists
from scipy.sparse.csgraph import shortest_path # for graph distances


# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class OrderOfFaces:
    """
    This class handles loading and processing facial image data for dimensionality
    reduction using the ISOMAP algorithm, with PCA as an optional comparison.

    Attributes:
    ----------
    images_path : str
        Path to the .mat file containing the image dataset.

    Methods:
    -------
    get_adjacency_matrix(epsilon):
        Returns the adjacency matrix based on a given epsilon neighborhood.

    get_best_epsilon():
        Returns the best epsilon for the ISOMAP algorithm, likely based on
        graph connectivity or reconstruction error.

    isomap(epsilon):
        Computes a 2D embedding of the data using the ISOMAP algorithm.

    pca(num_dim):
        Returns a low-dimensional embedding of the data using PCA.
    """

    def __init__(self, images_path='data/isomap.mat'):
        """
        Initializes the OrderOfFaces object and loads image data from the given path.

        Parameters:
        ----------
        images_path : str
            Path to the .mat file containing the facial images dataset.
        """

        self.images_path = images_path
        mat_data = self.load_data()
        self.data = self.process_data(mat_data)
        # calc dists now for reuse
        self.euc_dists = cdist(self.data, self.data, metric='euclidean')


    def get_adjacency_matrix(self, epsilon: float) -> np.ndarray:
        """
        Constructs the adjacency matrix using epsilon neighborhoods.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius within which points are considered connected.

        Returns:
        -------
        np.ndarray
            A 2D adjacency matrix (m x m) where each entry represents distance between
            neighbors within the epsilon threshold.
        """
        # calc pairwise distances (euclidean) for best epsilon
        euc_dists = self.euc_dists
        adj_matrix = euc_dists.copy()
        # zero out distances greater than eps
        adj_matrix = np.where(adj_matrix <= epsilon, adj_matrix, 0)
        return adj_matrix

    def get_best_epsilon(self, eps) -> float:
        """
        Heuristically determines the best epsilon value for graph connectivity in ISOMAP.

        Returns:
        -------
        float
            Optimal epsilon value ensuring a well-connected neighborhood graph.
        """
        euc_dists = self.euc_dists

        # rolling through eps options for all unique distances greater than 0
        # by using actual distance values connectivity should be better than random/arb values
        for eps in np.sort(np.unique(euc_dists[euc_dists > 0])):
            adj_matrix = self.get_adjacency_matrix(eps)
            # get shortest graphs distances (edge paths)
            g_dists = shortest_path(adj_matrix, directed=False)
        return eps


    def isomap(self, epsilon: float) -> np.ndarray:
        """
        Applies the ISOMAP algorithm to compute a 2D low-dimensional embedding of the dataset.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius for building the adjacency graph.

        Returns:
        -------
        np.ndarray
            A (m x 2) array where each row is a 2D embedding of the original data point.
        """
        """
        Per lecture slides I need to:
        Key idea: produce low dimensional representation which preserves “walkingdistance”
        over the data cloud (manifold)

        Find neighbors N^i of each data point, x^i, within distance eps and let A
        be the adjacency matrix recording neighbor Euclidean distance
        
        Find shortest path distance matrix D between each pairs of points, x^i
        and x^j, based on A

        Find low dimensional representation which preserves the distances
        information in D
        """
        # step 1: build weighted graph A using nearest neighbors
        # aka adjacency matrix
        adj_matrix = self.get_adjacency_matrix(epsilon)

        # step 2: get shortest pairwise distances matrix D
        D = shortest_path(adj_matrix, directed=False)
        D_2 = D ** 2

        # step 3: use centering matrix H to get C
        # H = I - (1/m) * 11^T, 11^T is outer prod of 1s vector
        m = D.shape[0]
        H = np.eye(m) - (1/m) * np.ones((m, m))
        # C = -0.5 * H * D^2 * H for centering matrix 
        C = -0.5 * H @ D_2 @ H

        # step 4: compute leading eigen vectors and eigen values of C
        eigvals, eigvecs = np.linalg.eigh(C)  # sym matrix so use eigh

        # step 4.5: arrange for leading eigen vecs and vals first
        # desceding
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx][:2] # get the top 2
        eigvecs = eigvecs[:, idx][:, :2] # get the top 2

        # step 5: get low dim representation
        # in final step of ISOMAP slide, the reduced rep shows sqrt of a diagonal eigval matrix
        L = np.diag(np.sqrt(eigvals))
        # multuply eigvecs by L to get final 2D rep
        X = eigvecs @ L  # 2 eigvecs by diag matrix of 2 eigvals

        return X # note that this was done with the top 2 leading eigvecs and vals

    def pca(self, num_dim: int) -> np.ndarray:
        """
        Applies PCA to reduce the dataset to a specified number of dimensions.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to project the data onto.

        Returns:
        -------
        np.ndarray
            A (m x num_dim) array representing the dataset in a reduced PCA space.
        """
        raise NotImplementedError("Not Implemented")
    
    def load_data(self):
        data = loadmat(self.images_path)
        return data
    
    def process_data(self, data):
        # shape info to better understand what the data looks like 
        for k, v in data.items():
            if hasattr(v, "shape"):
                print(f"{k} shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"{k} type={type(v)}")
        
        # process the matrix
        for k in data.keys():
            print(k)
        # orig shape is (4096, 698), need to transpose
        images = data['images'].T  # shape now (698, 4096)

        return images

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # algo class call
    order_of_faces = OrderOfFaces()

    # find best epsilon
    best_eps = order_of_faces.get_best_epsilon()

    # ISOMAP algo
    iso_map = order_of_faces.isomap(best_eps)


if __name__ == "__main__":
    main()