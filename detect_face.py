# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 03:07:07 2020

@author: imdevskp
"""

import sys
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage import data
from skimage.feature import Cascade

def show_detected_face(image, detected):
    '''takes an image and draw red rectangles around detected face'''
    plt.imshow(image)
    img_desc = plt.gca()
    plt.set_cmap('gray')
    plt.axis('off')
    
    for patch in detected:
        img_desc.add_patch(patches.Rectangle((patch['c'], patch['r']),
                                             patch['width'], patch['height'],
                                             fill=False, color='r', linewidth=2))
    plt.show()
    
def save_detected_face(image, detected):
    '''takes an image and draw red rectangles around detected face and saves the file'''
    plt.imshow(image)
    img_desc = plt.gca()
    plt.set_cmap('gray')
    plt.axis('off')
    
    for patch in detected:
        img_desc.add_patch(
            patches.Rectangle((patch['c'], patch['r']), 
                              patch['width'], patch['height'], 
                              fill=False, color='r', linewidth=2))
        
    save_img_name = 'detected_faces.jpg'
    plt.savefig(save_img_name, dpi=300, bbox_inches='tight', pad_inches=0)
    print('Image saved as', save_img_name)
    
# image source
img_src = sys.argv[1]

# read image
image = plt.imread(img_src)

# Load the trained file from the module root.
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade.
detector = Cascade(trained_file)

# Apply detector on the image
detected = detector.detect_multi_scale(img=image,
                                       scale_factor=1.2, step_ratio=1.2,
                                       min_size=(10, 10), max_size=(200, 200))

# No. of faces detected
print('No. of faces detected :', len(detected))

# Print patches for each faces
# for face in detected:
#     print(face)
    
# Show image with detected face marked
# show_detected_face(image, detected)

# Save image with detected face marked
save_detected_face(image, detected)