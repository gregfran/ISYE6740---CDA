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
        self.data = self.load_data()
        self.process_data(self.data)
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
        raise NotImplementedError("Not Implemented")

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