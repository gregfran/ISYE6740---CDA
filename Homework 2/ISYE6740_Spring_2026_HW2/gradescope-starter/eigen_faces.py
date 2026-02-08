# -----------------------------------------------------------------------------
# NOTE: This file consists of 2 classes

# 1. EigenFacesResult - This class should not be modified. Gradescope will use the output of run() 
# method in this format.
# 2. EigenFaces - This is class which will implement the eigen faces algorithm and return the results.  
# -----------------------------------------------------------------------------
import numpy as np
import os
from PIL import Image

# -----------------------------------------------------------------------------
# NOTE: This class should NOT be modified.
# Gradescope will depend on the structure of this class as defined. 
# -----------------------------------------------------------------------------
class EigenFacesResult:
    """    
    A structured container for storing the results of the EigenFaces computation.

    Attributes
    ----------
    subject_1_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 1.
        A plt.imshow(map['subject_1_eigen_faces'][0]) should display first in a eigen face for subject 1

    subject_2_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 2.
        A plt.imshow(map['subject_2_eigen_faces'][0]) should display first in a eigen face for subject 2

    s11 : float
        Projection residual of subject 1 test image on subject 1 eigenfaces.

    s12 : float
        Projection residual of subject 2 test image on subject 1 eigenfaces.

    s21 : float
        Projection residual of subject 1 test image on subject 2 eigenfaces.

    s22 : float
        Projection residual of subject 2 test image on subject 2 eigenfaces.
    """

    def __init__(
        self,
        subject_1_eigen_faces: np.ndarray,
        subject_2_eigen_faces: np.ndarray,
        s11: float,
        s12: float,
        s21: float,
        s22: float
    ):
        self.subject_1_eigen_faces = subject_1_eigen_faces
        self.subject_2_eigen_faces = subject_2_eigen_faces
        self.s11 = s11
        self.s12 = s12
        self.s21 = s21
        self.s22 = s22
        
# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class EigenFaces:
    """
    This class handles loading facial images for two subjects, computing eigenfaces
    via PCA, and evaluating projection residuals for test images.

    Methods
    -------
    run():
        Computes the eigenfaces for each subject and the projection residuals for test images.
    """

    def __init__(self, images_root_directory="data/yalefaces"):
        """
        Initializes the EigenFaces object and loads all relevant facial images from the specified directory.

        Parameters
        ----------
        images_root_directory : str
            The path to the root directory containing subject images.
        """
        self.img_dir = images_root_directory

        # load the train images for each subject
        self.subject_1_tr_images = self.load_data(1)
        self.subject_2_tr_images = self.load_data(2)

        # load the test images for each subject
        self.subject_1_te_image = self.load_data(1, is_test=True)
        self.subject_2_te_image = self.load_data(2, is_test=True)

    def calc_eigenfaces(self, imgs, eigs=6):
        # dimensions
        n_imgs, h, w = imgs.shape

        # img to vectors
        img_vecs = imgs.reshape(n_imgs, h*w)

        # avg face
        avg_face = np.mean(img_vecs, axis=0)

        # center data
        img_vecs_centered = img_vecs - avg_face

        # calc C (covariance matrix)
        C = np.cov(img_vecs_centered, rowvar=False)

        # get eigenvals and vecs from C
        eigenvals, eigenvecs = np.linalg.eigh(C)

        # get top eigenfaces
        idx = np.argsort(eigenvals)[::-1][:eigs]
        top_eigenvecs = eigenvecs[:, idx]

        eigenfaces = top_eigenvecs.T.reshape(eigs, h, w)

        return eigenfaces, avg_face, top_eigenvecs.T
    
    def load_data(self, subject_no, is_test=False):
        img_dir = self.img_dir
        imgs = []

        # roll thru images in file directory
        for subj in os.listdir(img_dir):
            # if binary is false, load train images
            if not is_test:
                if (f"subject{str(subject_no).zfill(2)}") in subj and not "test" in subj:
                    subj_path = os.path.join(img_dir, subj)
                    img = Image.open(subj_path).convert('L')
                    img_arr = np.array(img)
                    img_arr_ds = self.img_ds(img_arr)
                    imgs.append(img_arr_ds)

            # if binary is true, load test image
            else:
                if (f"subject{str(subject_no).zfill(2)}") in subj and "test" in subj:
                    subj_path = os.path.join(img_dir, subj)
                    img = Image.open(subj_path).convert('L')
                    img_arr = np.array(img)
                    img_arr_ds = self.img_ds(img_arr)
                    return img_arr_ds
            
        return np.array(imgs)
        
    def proj_res(self, test_img, eigenvecs, avg_face):

        # test img vectorization
        test_vec = test_img.flatten()
        avg_vec = avg_face.flatten()

        # center test img
        test_vec_centered = test_vec - avg_vec

        # projection
        proj = np.zeros_like(test_vec_centered)

        for e in eigenvecs:
            coeff = np.dot(test_vec_centered, e)
            proj += coeff * e
        
        # proj residual
        p_res = np.linalg.norm(test_vec_centered - proj) ** 2

        return p_res

    def img_ds(self, img_arr, factor=4):
        # downsample by factor
        # gradescope is giving me issues due to the size
        # my downsampling method is (6, 61, 80) and should be (6, 60, 80)
        h, w = img_arr.shape
        h_ds = h // factor # floor divisions
        w_ds = w // factor
        # limit downsample to new dimension targets
        return img_arr[:h_ds*factor:factor, :w_ds*factor:factor] # downsample by only taking every factor-th pixel

    def run(self) -> EigenFacesResult:
        """
        Computes eigenfaces for both subjects and projection residuals
        for test images using those eigenfaces.

        Returns
        -------
        EigenFacesResult
            Object containing eigenfaces and residuals for both subjects.
        """
        # get eigenfaces
        eigenfaces_1, avg_face_1, eigenvecs_1 = self.calc_eigenfaces(self.subject_1_tr_images)
        eigenfaces_2, avg_face_2, eigenvecs_2 = self.calc_eigenfaces(self.subject_2_tr_images)

        # proj residuals part of return
        return EigenFacesResult(
            subject_1_eigen_faces=eigenfaces_1,
            subject_2_eigen_faces=eigenfaces_2,
            s11=self.proj_res(self.subject_1_te_image, eigenvecs_1, avg_face_1),
            s12=self.proj_res(self.subject_2_te_image, eigenvecs_1, avg_face_1),
            s21=self.proj_res(self.subject_1_te_image, eigenvecs_2, avg_face_2),
            s22=self.proj_res(self.subject_2_te_image, eigenvecs_2, avg_face_2)
        )