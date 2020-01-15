"""
Manil Bastola
3D CTScan Data class. Inspired from 2D CTScan Image saver @ https://github.com/swethasubramanian/LungCancerDetection/blob/master/src/data/create_images.py 
"""

import numpy as np
import pandas as pd

import os
import glob

from PIL import Image
import SimpleITK as sitk


base_dir = "/media/mbastola/Transcend/LUNA/data/"
candidates_file = 'candidates.csv'

class CTSData(object):
    """
       Reads .mhd header data, crops images around annotations and generates/saves cropped image
	"""
    def __init__(self, filename = None, coords = None, path = None):
        """
        filename: .mhd filename
        coords: coordinates to crop around
        path: path to directory with all the raw data
        """
        self.filename = filename
        self.coords = coords
        self.metadata = None
        self.eps = 10
        self.images = []
        self.path = path

    def read_images(self):
        path = glob.glob(self.path + self.filename + '.mhd')
        self.metadata = sitk.ReadImage(path[0])
        return sitk.GetArrayFromImage(self.metadata)

    def gen_images(self, width = 64):
        images = self.read_images()
        images = self.get_annotation_roi(images, width)
        images = self.normalize_intensity(images)
        self.images = images
    
    def get_annotation_roi(self, image, width):
        """
        Returns cropped image of requested dimension
        """
        origin = self.metadata.GetOrigin()
        resolution = self.metadata.GetSpacing()
        x,y,z = [np.absolute(self.coords[j]-origin[j])/resolution[j] for j in range(len(self.coords))]
        roi = image[:, int(y-width/2):int(y+width/2), int(x-width/2):int(x+width/2)]
        
	#subsample uniformly for number = eps many slices
        slen = len(roi[int(z-self.eps/2):int(z+self.eps/2)])
        k = 0
        while (slen < self.eps):
            k += 2
            slen = len(roi[int(z- self.eps/2 - k/2):int(z+self.eps/2+k/2)])
        roi = roi[int(z-self.eps/2 - k/2):int(z+self.eps/2+k/2)]
        roi = roi[:self.eps]
        return roi

    def normalize_intensity(self, npzarray):
        """
        Houndsunits to grayscale units
        """
        maxHU = 400.0
        minHU = -1000.0
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray
    
    def save_image(self, filename):
        """
        Save annotation cropped CT image
        """
        if len(self.image) == 0:
            self.gen_images()
        Image.fromarray(self.images[int(self.eps/2)]*255).convert('L').save(filename)

    def get_image_slice(self):
        if len(self.images) == 0:
            self.gen_images()
        return np.ravel(self.images[int(self.eps/2)])

    def get_images(self, width=64):
        if len(self.images) == 0:
            self.gen_images(width)
        data = self.images.reshape(-1,width*width)
        return data
