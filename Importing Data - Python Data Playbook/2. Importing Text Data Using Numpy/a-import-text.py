# How to read a plain text file
which_file = "Creative Commons Attribution-ShareAlike 3.0 Unported.txt"
license_file = open(which_file, mode='r')
license_name = license_file.readline()
print(license_name)
license = license_file.read()
license_file.close()
print(license)

# Same as above, but using a context manager to avoid having to explicitly close the file
with open (which_file, 'r') as file:
    print(file.read())