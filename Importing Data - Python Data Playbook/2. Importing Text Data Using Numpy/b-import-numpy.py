import numpy as np


# Import a list using np.array
sample_array = np.array([0,0,7])
type(sample_array)

# Import a NumPy text file
# cat badges-five-numpy.txt
badges_saved_np = np.loadtxt('badges-five-numpy.txt')
badges_saved_np
badges_saved_np.size
badges_saved_np.shape

# Import a comma separated file, required to specify delimiter
# cat badges-five.txt
badges_comma = np.loadtxt('badges-five.txt') # error
badges_comma = np.loadtxt('badges-five.txt', delimiter=',')
badges_comma

# Import a comma separated file that has a header row
# cat badges-five-header.txt
badges_header = np.loadtxt('badges-five-header.txt', delimiter=',') # error
badges_header = np.loadtxt('badges-five-header.txt', delimiter=',', skiprows=1) 
badges_header

# Import a comma separated file with header row and convert values to string
badges_str = np.loadtxt('badges-five-header.txt', delimiter=',', skiprows=1, dtype=str) 

# Import and apply a function to one of the values
def Increase(the_id):
    return int(the_id) + 1000

badges_increased = np.loadtxt('badges-five.txt', delimiter=',', dtype=int, converters={0: Increase})

# Import with a missing value, but still with loadtxt which raises an error
# cat badges-five-missing-value.txt
badges_missing_value = np.loadtxt('data\\badges-five-missing-value.txt', delimiter=',', skiprows=1)

# Import with a missing value using genfromtxt
badges_missing_value = np.genfromtxt('badges-five-missing-value.txt', delimiter=',')
badges_missing_value

# We need to skip the header, and then we explore our data with size and shape
badges_missing_value = np.genfromtxt('badges-five-missing-value.txt', delimiter=',', skip_header=1)
badges_missing_value

# And you can specify what to do with missing values, i.e. fill with another value
badges_missing_value = np.genfromtxt('badges-five-missing-value.txt', delimiter=',', skip_header=1, filling_values=0)

