import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates


###################################################################################################################################
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

###################################################################################################################################
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
    
    # Q6 - for this implementation, the query and control points are reshaped to (N, 1, 3) and (1, M, 3) respectively
    # broadcasting is used to compute all the pairwise distances and returns a shape of (N, M, 3) 
    # distances are squared and added along axis from which 
    # the Gaussian kernel formula is then applied to the entire matrix via numpy array operations. 

    # Q7 - sigma controls the spatial extent of each control point's influence. 
    # a smaller signa results in localised deformations concentrated around the control points as opposed to a large sigma that creates global defomations


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
    
    # Q1 -- no, the polynomial part is not needed for this implementation. 
    # for some RBFs, the polynomial is needed to satisfy "conditional positive definiteness" to guaraantee solvability of the kernel matrix.
    # since the Gaussian kernel is positive definite, the kernel matrix K is always invertible. 
    # therefore, there is no need to include the polynomial stabiliser term that traditional RBF methods require.

    # Q2 - linear algebra formula for spline fitting:
    # [K + lambda * W^(-1)] * alpha = q_k    [equation 12, Fornefett 2001]
    # solution: alpha = [K + lambda * W^(-1)]^(-1) @ q_k
    # where: K = NxN kernel matrix, K[i,j] = phi(||p_i - p_j||) using Gaussian kernel
    # alpha = coefficients to solve for
    # q_k = target displacements -- from equation 16
    # lambda = regularisation parameter
    # W = diagonal weight matrix
    # 
    # u(x) = x + sum(alpha * phi(||x - p||)) [equation 16, Fornefett 2001]
    # this shows the RBF is applied with identity transformation, therefore solve for (q_k - p_k) rather than absolute target positions
    # in code: alpha = np.linalg.solve(K + lambda * I, displacements)

    # Q3 - the optimal linear algebra algorithm is the Cholesky decomposition. 
    # for compact-support RBFs, K is symmetric positive definite and sparse and Cholesky peforms ~2x faster than LU decomposition.
    # although np.linalg.solve uses LU decomposition internally, scipy.linalg.solve with assume_a='pos' would use Cholesky.


    # FUNCTION to apply fitted spline to transformation
    def evaluate(self, query_points, control_points, coeffs, sigma=20.0):
        # query_points = the pixel points to transform
        # control_points = control used in fit
        # calculate how close query points are to control
        K_eval = self.kernel_gaussian(query_points, control_points, sigma)
        calc_displacements = np.dot(K_eval, coeffs)
        return query_points + calc_displacements

    # Q4 - the control points are landmarks (p_i) where the RBFs are centered during fitting.
    # we cannot choose different points at evaluation since the transformation is a weighted sum of RBFs around the specific landmarks.
    # changing the control points would mean that the alpha weights are not applied, producing a different transformation than fitted. 

    # Q5 - we do not need lambda at evaluation stage as it is only used when solving for alpha coefficients.
    # once the spline is fitted, the transformation only depends on the distances and weights, which already incorporates regularisation.


###################################################################################################################################
# CLASS FreeFormDeformation
class FreeFormDeformation:  # for grid generation and image warping
    def __init__(self, nx, ny, nz, min_max_x, min_max_y, min_max_z):
        # number of control points in each direction
        self.nx, self.ny, self.nz, = nx, ny, nz
        self._generate_grid(min_max_x, min_max_y, min_max_z)  # range in mm

    @classmethod
    def from_image(cls, image_obj, nx, ny, nz):
        (min_x, max_x), (min_y, max_y), (min_z,
                                         max_z) = image_obj.get_physical_mm()  # retrieving mm for image
        return cls(nx, ny, nz, (min_x, max_x), (min_y, max_y), (min_z, max_z))

    def _generate_grid(self, range_x, range_y, range_z):  # meshgrid of control points
        # evenly spaced nx points
        x = np.linspace(range_x[0], range_x[1], self.nx)
        y = np.linspace(range_y[0], range_y[1], self.ny)
        z = np.linspace(range_z[0], range_z[1], self.nz)

        Z, Y, X = np.meshgrid(z, y, x, indexing='ij')  # ij ensures z, y, z

        self.source_control_points = np.column_stack(
            [Z.ravel(), Y.ravel(), X.ravel()])  # flatten to a list of (M, 3)
        # initialise target as the same as source
        self.target_control_points = self.source_control_points.copy()

    # generates randomly displaced target control points
    def random_transform_generator(self, strength=0.5):
        # generate random noise between -1 and 1
        noise = np.random.uniform(-1, 1, size=self.source_control_points.shape)

        mag = 10.0 * strength
        self.target_control_points = self.source_control_points + \
            noise * mag  # scale noise against magnitude
        return self.target_control_points
    
    # Q8 - in terms of a resonable approach to randomly displace control points, first we would consider a Gaussian distrubtion.
    # a Gaussian distribution produces smoother deformations as opposed to a uniform distribution - helps represent natural patient variability better.
    # to avoid image folding/tearing, the displacement scale should be relative to control point spacing and not use absolute values
    # it is important to consider resolution impacts as flat displacement produces different results at different resolutions.

    # Q9 - although smooth control point displacements produce smooth deformations, it is not guaranteed to be biophysically plausible.
    # for true biophysical plausibility, voxel neighbourhood relationships need to be maintained to preserve topolgy.
    # the Jacobian determinant must also be positive everywhere to prevent image folding/tearing
    # also, since different tissue types deform differently, there must be qualitative analysis since RBF treats everything equally. 

    def warp_image(self, image_obj, rbf_spline, sigma=20.0):
        data = image_obj.data
        spacing = image_obj.spacing  # (z,y,x)
        shape = data.shape

        # grid of all indices - voxel coordinates
        z_idx = np.arange(shape[0])
        y_idx = np.arange(shape[1])
        x_idx = np.arange(shape[2])

        grid_z, grid_y, grid_x = np.meshgrid(
            z_idx, y_idx, x_idx, indexing='ij')

        coordinates_indices = np.column_stack(
            [grid_z.ravel(), grid_y.ravel(), grid_x.ravel()])
        coordinates_mm = coordinates_indices * \
            np.array(spacing)  # convert indices to physical space

        # fit
        coefficients = rbf_spline.fit(
            self.source_control_points, self.target_control_points, sigma=sigma)

        # apply deformation field to the physical coordinates
        new_coordinates_mm = rbf_spline.evaluate(
            coordinates_mm, self.source_control_points, coefficients, sigma)
        new_coordinates_indices = new_coordinates_mm / \
            np.array(spacing)  # convert back to voxel indices

        # transpose as (3 x N) is needed
        sampling_coordinates = new_coordinates_indices.T

        warped_data = map_coordinates(
            data, sampling_coordinates, order=1, mode='nearest')  # interpolate pixel values
        return warped_data.reshape(shape)
    
    # Q10 - to compute a warped 3D image, first, generate the meshgrid of all voxel indices using np.arange and np.meshgrid
    # then flatten and stack indices to Nx3 array using column_stack and ravel
    # convert voxel indices to physical coordinates by multiplying by spacing and fit RBF spline using control points to get coefficients (alpha)
    # evaluate the fitted spline at all physical coordinates to get transformation and convert back to voxel indices (by division)
    # transpose coordinates to (3, N) required by map_coordinates and interpolate original image at coordinates
    # (order=1 for linear interpolation, mode='nearest' for boundary handling) and finally reshape interpolated values back to 3D original volume shape.

    # random deformation
    def random_transform(self, image_obj, rbf_spline, strength=0.5, sigma=20.0):
        self.random_transform_generator(strength)
        return self.warp_image(image_obj, rbf_spline, sigma)

###################################################################################################################################
###################################################################################################################################
if __name__ == "__main__":
    print("-- starting task1 -- ")

    try:
        img_object = Image3D("image_train00.npy")
        print("data loaded")
    except:
        print("warning: image not found. using dummy data")
        img_object = Image3D(data=np.zeros((10, 64, 64)))
        img_object.data[2:8, 20:44, 20:44] = 1

    rbf = RBFSpline()

    # 4 x 4 x 4 grid of control points
    ffd = FreeFormDeformation.from_image(img_object, 4, 4, 4)
    z_depths = np.linspace(0, img_object.shape[0]-1, 5, dtype=int)

    fig_joint, axes_joint = plt.subplots(10, 5, figsize=(15, 30))
    fig_joint.suptitle(
        "10 random transformations (rows) at 5 z_depths (columns)", fontsize=16)

    for i in range(1, 11):
        print(f"processing {i}/10")
        warped_volume = ffd.random_transform(
            img_object, rbf, strength=0.5, sigma=20.0)

        for j, z in enumerate(z_depths):
            ax = axes_joint[i-1, j]
            ax.imshow(warped_volume[z, :, :], cmap='gray')
            ax.axis("off")

            if i == 1:  # top row
                ax.set_title(f"slice z={z}")
            if j == 0:  # left column
                ax.text(-1, 32, f"{i}", fontsize=12, rotation=90)

        # figure with 5 individiual subplots
        plt.figure(figsize=(15, 3))
        plt.suptitle(f"random transformation {i}", fontsize=14)

        for idx, z in enumerate(z_depths):
            plt.subplot(1, 5, idx + 1)
            plt.imshow(warped_volume[z, :, :], cmap='gray')
            plt.title(f"slice z={i}")
            plt.axis("off")

        indiv_filename = f"transform_{i:02d}_5slices.png"
        plt.savefig(indiv_filename)
        plt.close()

    fig_joint.tight_layout(rect=[0, 0.02, 1, 0.97])
    joint_name = "combined_transforms.png"
    fig_joint.savefig(joint_name)
    plt.close(fig_joint)
    print("saved 10 transformation plots")

    # Q11 - the combined_transforms.png plot supports the idea explored in Q9.
    # the smooth RBF interpolation produces visually smooth deformations but are not guaranteed biophysically plausible.
    # the anatomical structures such as bladder and prostate remain recognisable, sugggesting the topology is mostly preserved at strength=0.5
    # however, some rows show tissue boundaries to be unnaturally stretched or compressed (rows 4,9,10 at z=31).
    # the edge slices show more visible warping artefacts than central slices 
    # and there are no obvious image folding/tearing using these parameter settings.

    print("-- visualising parameter changes --")

    print("-- varied kernel width (sigma) --")
    view_z = img_object.shape[0] // 2
    ffd_std = FreeFormDeformation.from_image(img_object, 4, 4, 4)
    vol_sigma_low = ffd_std.random_transform(img_object, rbf, strength=0.5, sigma=5.0) #lower sigma = local influence
    vol_sigma_high = ffd_std.random_transform(img_object, rbf, strength=0.5, sigma=50.0) #higher sigma = global influence ie stretched out and smoothed

    #plot comparison
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(vol_sigma_low[view_z,: ,:], cmap='gray')
    plt.title("sigma=5.0 (local influence)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(vol_sigma_high[view_z, :, :], cmap='gray')
    plt.title("sigma=50.0 (global influence)")
    plt.axis("off")

    plt.savefig("param_sigma_variation.png")
    plt.close()
    print("saved sigma variation png")

    print("-- varied strength --")
    ffd_std = FreeFormDeformation.from_image(img_object, 4, 4, 4)
    vol_strength_lower = ffd_std.random_transform(
        img_object, rbf, strength=0.1, sigma=20.0)  # lower strength
    vol_strength_higher = ffd_std.random_transform(
        img_object, rbf, strength=1.0, sigma=20.0)  # higher strength

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(vol_strength_lower[view_z, :, :], cmap='gray')
    plt.title("strength = 0.1")
    plt.subplot(1, 2, 2)
    plt.imshow(vol_strength_higher[view_z, :, :], cmap='gray')
    plt.title("strength = 1.0")
    plt.savefig("param_strength_variation.png")
    plt.close()
    print("saved strength variation png")

    print("-- varied grid size --")
    ffd_dense = FreeFormDeformation.from_image(img_object, 8, 8, 8) #denser grid for higher frequency deformation
    vol_dense = ffd_dense.random_transform(img_object, rbf, strength=0.5, sigma=20.0)

    plt.figure(figsize=(5,5))
    plt.subplot(1, 2, 1)
    plt.imshow(vol_sigma_low[view_z, :, :], cmap='gray')
    plt.title("grid (4x4x4)")
    plt.subplot(1, 2, 2)
    plt.imshow(vol_dense[view_z, :, :], cmap='gray')
    plt.title("denser grid (8x8x8)")
    plt.savefig("grid_variation.png")
    plt.close()
    print("saved grid variation png")

    # Q12 - by varying the value of sigma, we can see that a higher value (50.0) results in smoother, more global deformation
    # where structures appear more uniformly shifted, as opposed to sharper deformations visible around anatomical edges.
    # increasing the strength from 0.1 to 1.0 results in more noticeable warping. the structures are visibly displaced but still recognisable.
    # when testing a range of strengths, higher values > 1.0 lead to image folding/tearing.
    # additionally, adding more control points helped achieve fine-grained control but the deformations appeared similar 
    # due to the existing values of strength and sigma.

    print("all tasks complete")
