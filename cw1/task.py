import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

# CLASSES
# 3D image class to handle medical images with information


class Image3D:
    # filepath = path to .npy file
    # data = extra data that might be included
    # spacing = voxel spacing in mm (z,y,x)
    def __init__(self, filepath=None, data=None, spacing=None):
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


#################################################################################
if __name__ == "__main__":
    print("testing 3d_image class with npy image")

    filename = "image_train00.npy"

    try:
        image_obj = Image3D(filepath=filename)
        print(f"success! loaded file properly")
        print(f"image shape(z,y,x) = {image_obj.shape}")

        print(f"voxel spacing (z,y,x): {image_obj.spacing}")

        (x_range, y_range, z_range) = image_obj.get_physical_mm()
        print(f"x range: {x_range}")
        print(f"y range: {y_range}")
        print(f"z range: {z_range}")

        middle_slice_index = image_obj.shape[0] // 2
        # give the slice data at the middle
        slice_data = image_obj.data[middle_slice_index, :, :]

        # plotting
        plt.figure(figsize=(6, 6))
        plt.imshow(slice_data, cmap='gray')
        plt.title(f"middle slice")
        plt.axis("off")
        plt.show()

    except FileNotFoundError:
        print("couldnt find filename in directory")

    except Exception as e:
        print("an unexpected error occurred")
