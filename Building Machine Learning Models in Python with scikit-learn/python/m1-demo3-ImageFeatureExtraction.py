
# coding: utf-8

# # Extracting Features from Images
# ##### Obtaining pixel intensities for image and transforming those into usable features

# #### Installation of opencv may be required
# 
# OpenCV == Open Source Computer Vision Library

# In[1]:

get_ipython().system(u'pip install opencv-python')


# In[2]:

import cv2


# #### Load an image from which to extract RGB pixel intensities

# In[3]:

# Image with dimensions 173x130
imagePath = '../data/dog.jpg'

image = cv2.imread(imagePath)


# #### View image using matplotlib

# In[4]:

get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt

plt.imshow(image)


# #### Image array contains RGB values for each pixel in the image

# In[5]:

image.shape


# In[6]:

image


# #### Each pixel has RGB intensities

# In[7]:

image[0][0]


# #### Scale this image to a smaller size

# In[8]:

size=(32, 32)
resized_image_feature_vector = cv2.resize(image, size)


# In[9]:

plt.imshow(resized_image_feature_vector)


# In[10]:

resized_image_feature_vector.shape


# In[11]:

resized_image_feature_vector


# #### Image array can be flattened into a one-dimensional array

# In[12]:

resized_flattened_image_feature_vector = resized_image_feature_vector.flatten()


# In[13]:

len(resized_flattened_image_feature_vector)


# In[14]:

image_grayscale = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE )


# In[15]:

plt.imshow(image_grayscale)


# In[16]:

image_grayscale.shape


# In[17]:

image_grayscale


# In[18]:

import numpy as np

expanded_image_grayscale = np.expand_dims(image_grayscale, axis=2)
expanded_image_grayscale.shape


# In[19]:

expanded_image_grayscale


# In[ ]:



