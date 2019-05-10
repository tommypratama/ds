"""pyplt-demo.py: A small demo script showing the pyplot API

Make sure anaconda is setup correctly -- or at least a virtual
environment with pandas and matplotlib installed."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = range(-10,11)
y = np.power(x, 3)
plt.plot(x, y, 'g')
plt.savefig('demo.png')
plt.title('Third power of x')

y2 = 13*np.power(x,2) - y - 1000
plt.plot(x, y2, 'r')

# create a new figure
plt.figure()
df = pd.read_csv('water.csv')
plt.plot(df['height'])

# show the figure
plt.show()
