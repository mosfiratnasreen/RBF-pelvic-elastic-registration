import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

### CLASSES
# 3D image class to handle medical images with information
class 3D_image: 
    #filepath = path to .npy file 
    #data = extra data that might be included
    #spacing = voxel spacing in mm (z,y,x)
    def __init__(self, filepath=None, data=None, spacing=None): 
        if filepath:
            self.data = np.load(filepath) #load data using numpy
        elif data is not None:
            self.data = data #incase raw data is provided instead
        else:
            raise ValueError("file path or data must be provided")
        
        #we want (0.5 x 0.5) for x,y and 2.0mm for z
        if spacing:
            self.spacing = spacing #if specified
        else:
            self.spacing = (2.0, 0.5, 0.5) #default
        
        self.shape = self.data.shape #store data dimensions
    
    