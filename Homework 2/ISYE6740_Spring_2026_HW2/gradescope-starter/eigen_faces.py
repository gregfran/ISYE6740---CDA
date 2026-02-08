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
        self.subject_1_te_image = self.load_data(1, test=True)
        self.subject_2_te_image = self.load_data(2, test=True)

    def calc_eigenfaces():

        return
    
    def load_data(self, subject_no, is_test=False):
        img_dir = self.img_dir
        imgs = []

        # roll thru images in file directory
        for subj in os.listdir(img_dir):
            # if binary is false, load train images
            if not is_test:
                if subj.contains(f"subject{str(subject_no).zfill(2)}") and not subj.contains("test"):
                    subj_path = os.path.join(img_dir, subj)
                    img = Image.open(subj_path).convert('L')
                    img_arr = np.array(img)
                    img_arr_ds = self.img_ds(img_arr)
                    imgs.append(img_arr_ds)

            # if binary is true, load test image
            else:
                if subj.contains(f"subject{str(subject_no).zfill(2)}") and subj.contains("test"):
                    subj_path = os.path.join(img_dir, subj)
                    img = Image.open(subj_path).convert('L')
                    img_arr = np.array(img)
                    img_arr_ds = self.img_ds(img_arr)
                    return img_arr_ds
            
            return np.array(imgs)


    def img_ds(self, img_arr, factor=4):
        # downsample by factor
        return img_arr[::factor, ::factor] # downsample by only taking every factor-th pixel

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
        eigenfaces_1 = self.calc_eigenfaces()

        return EigenFacesResult(
            subject_1_eigen_faces=eigenfaces_1,
            subject_2_eigen_faces=eigenfaces_2,
            s11=projection_residual_s11,
            s12=projection_residual_s12,
            s21=projection_residual_s21,
            s22=projection_residual_s22
        )