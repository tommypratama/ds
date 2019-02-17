import xml.etree.ElementTree as ET

# Parse into an ElementTree
tree = ET.parse('users-100.xml')
tree

# Get the root, children, and review one Element
users_root = tree.getroot()
users_root.tag
users_root.getchildren()
len(list(users_root.getchildren()))
users_root[0]
users_root[0].tag
users_root[0].attrib

