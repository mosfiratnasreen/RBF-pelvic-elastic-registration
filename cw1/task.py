import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

# CLASSES

# CLASS 3D image to handle medical images with information
class Image3D:
    def __init__(self, filepath=None, data=None, spacing=None):
        # filepath = path to .npy file
        # data = extra data that might be included
        # spacing = voxel spacing in mm (z,y,x)
        if filepath:
            self.data = np.load(filepath)  # load data using numpy
        elif data is not None:
            self.data = data  # incase raw data is provided instead
        else:
            raise ValueError("file path or data must be provided")

        # we want (0.5 x 0.5) for x,y and 2.0mm for z
        if spacing:
            self.spacing = spacing  # if specified
        else:
            self.spacing = (2.0, 0.5, 0.5)  # default

        self.shape = self.data.shape  # store data dimensions

    # FUNCTION to calculate the physical size of the image in mm for x,y,z
    def get_physical_mm(self):
        # depth -- number of slices x thickness
        z_max = self.shape[0] * self.spacing[0]
        # height -- rows x height of pixel
        y_max = self.shape[1] * self.spacing[1]
        x_max = self.shape[2] * self.spacing[2]  # width -- columns x width
        return (0, x_max), (0, y_max), (0, z_max)

# CLASS RBFSpline
class RBFSpline:
    def __init__(self):
        pass

    # FUNCTION to calculate gaussian kernel values between query and control points
    def kernel_gaussian(self, query_points, control_points, sigma=10.0):
        # query_points = np.ndarray (N,3)
        # control_points = np.ndarray (M,3)
        # sigma = float Gaussian width
        # calculate the difference between query and control point
        # (N, 1, 3) - (1, M, 3) -> (N, M, 3)
        difference = query_points[:, np.newaxis, :] - \
            control_points[np.newaxis, :, :]

        # sq and sum differences
        diff_sq = np.sum(difference**2, axis=2)  # (N, M)

        # apply Gaussian formula
        K = np.exp(-diff_sq / (sigma ** 2))
        return K

    # FUNCTION to fit the spline
    def fit(self, source_points, target_points, lambda_param=0.01, sigma=20.0):
        # source_points (M,3)
        # target_points (M,3)
        # sigma == kernel width
        M = source_points.shape[0]  # number of control points
        # creating kernel matrix
        K = self.kernel_gaussian(source_points, source_points, sigma)

        K_reg = K + lambda_param * np.eye(M)  # adding regularisation

        displacements = target_points - source_points  # vector D in the equation

        # solve to find the coefficients C
        self.coeffs = np.linalg.solve(K_reg, displacements)
        return self.coeffs

    # FUNCTION to apply fitted spline to transformation
    def evaluate(self, query_points, control_points, coeffs, sigma=20.0):
        # query_points = the pixel points to transform
        # control_points = control used in fit
        # calculate how close query points are to control
        K_eval = self.kernel_gaussian(query_points, control_points, sigma)
        calc_displacements = np.dot(K_eval, coeffs)
        return query_points + calc_displacements

# CLASS FreeFormDeformation
class FreeFormDeformation: #for grid generation and image warping
    def __init__(self, nx, ny, nz, min_max_x, min_max_y, min_max_z):
        self.nx, self.ny, self.nz, = nx, ny, nz #number of control points in each direction
        self._generate_grid(min_max_x, min_max_y, min_max_z) #range in mm

    @classmethod
    def from_image(cls, image_obj, nx, ny, nz):
       (min_x, max_x), (min_y, max_y), (min_z, max_z) = image_obj.get_physical_mm() #retrieving mm for image
       return cls(nx, ny, nz, (min_x, max_x), (min_y, max_y), (min_z, max_z))
    
    def _generate_grid(self, range_x, range_y, range_z): #meshgrid of control points
        x = np.linspace(range_x[0], range_x[1], self.nx) #evenly spaced nx points
        y = np.linspace(range_y[0], range_y[1], self.ny)
        z = np.linspace(range_z[0], range_z[1], self.nz)

        Z, Y, X = np.meshgrid(z, y, x, indexing='ij') #ij ensures z, y, z

        self.source_control_points = np.column_stack([Z.ravel(), Y.ravel(), X.ravel()]) #flatten to a list of (M, 3)
        self.target_control_points = self.source_control_points.copy() #initialise target as the same as source

    def random_transform_generator(self, strength=0.5): #generates randomly displaced target control points
        noise = np.random.uniform(-1, 1, size=self.source_control_points.shape) #generate random noise between -1 and 1

        mag = 10.0 * strength
        self.target_control_points = self.source_control_points + noise * mag #scale noise against magnitude






















###########################################################################################################################
if __name__ == "__main__":
    # print("testing 3d_image class with npy image")

    # filename = "image_train00.npy"

    # try:
    #     image_obj = Image3D(filepath=filename)
    #     print(f"success! loaded file properly")
    #     print(f"image shape(z,y,x) = {image_obj.shape}")

    #     print(f"voxel spacing (z,y,x): {image_obj.spacing}")

    #     (x_range, y_range, z_range) = image_obj.get_physical_mm()
    #     print(f"x range: {x_range}")
    #     print(f"y range: {y_range}")
    #     print(f"z range: {z_range}")

    #     middle_slice_index = image_obj.shape[0] // 2
    #     # give the slice data at the middle
    #     slice_data = image_obj.data[middle_slice_index, :, :]

    #     # plotting
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(slice_data, cmap='gray')
    #     plt.title(f"middle slice")
    #     plt.axis("off")
    #     plt.show()

    # except FileNotFoundError:
    #     print("couldnt find filename in directory")

    # except Exception as e:
    #     print("an unexpected error occurred")

    print("testing RBFspline")
    # simple (random) source points
    source_pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    # 5mm to the right target points
    target_pts = np.array([[0.0, 0.0, 0.0], [15.0, 0.0, 0.0]])

    rbf = RBFSpline()
    coefficients = rbf.fit(source_pts, target_pts, sigma=10.0)
    print("spline fitted - coefficients calculated")

    test = np.array([[10.0, 0.0, 0.0]])
    result = rbf.evaluate(test, source_pts, coefficients, sigma=10.0)

    print(f"original point {test[0]}")
    print(f"transformed point {result[0]}")
    print("expected [15.0, 0.0, 0.0]")
