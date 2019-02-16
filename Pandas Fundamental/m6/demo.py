# -*- coding: utf-8 -*-
 
import pandas as pd
import os

df = pd.read_pickle(os.path.join('..', 'data_frame.pickle'))

# Simplest default plot
acquisition_years = df.groupby('acquisitionYear').size()
acquisition_years.plot()


import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True,
                 'axes.titlepad': 20})

fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)
acquisition_years.plot(ax=subplot)
fig.show()

# Add axis labels
fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)
acquisition_years.plot(ax=subplot)
subplot.set_xlabel("Acquisition Year")
subplot.set_ylabel("Artworks Acquired")
fig.show()



# Increase ticks granularity
fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)
acquisition_years.plot(ax=subplot)
subplot.set_xlabel("Acquisition Year")
subplot.set_ylabel("Artworks Acquired")
subplot.locator_params(nbins=40, axis='x')
fig.show()


# Rotate X ticks
fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)
acquisition_years.plot(ax=subplot, rot=45)
subplot.set_xlabel("Acquisition Year")
subplot.set_ylabel("Artworks Acquired")
subplot.locator_params(nbins=40, axis='x')
fig.show()



# Add log scale
fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)
acquisition_years.plot(ax=subplot, rot=45, logy=True)
subplot.set_xlabel("Acquisition Year")
subplot.set_ylabel("Artworks Acquired")
subplot.locator_params(nbins=40, axis='x')
fig.show()



# Add grid
fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)
acquisition_years.plot(ax=subplot, rot=45, logy=True, grid=True)
subplot.set_xlabel("Acquisition Year")
subplot.set_ylabel("Artworks Acquired")
subplot.locator_params(nbins=40, axis='x')
fig.show()


# Set fonts
title_font = {'family': 'source sans pro',
        'color':  'darkblue',
        'weight': 'normal',
        'size': 20,
        }
labels_font = {'family': 'consolas',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }



fig = plt.figure()
subplot = fig.add_subplot(1, 1, 1)
acquisition_years.plot(ax=subplot, rot=45, logy=True, grid=True)
subplot.set_xlabel("Acquisition Year", fontdict=labels_font, labelpad=10)
subplot.set_ylabel("Artworks Acquired", fontdict=labels_font)
subplot.locator_params(nbins=40, axis='x')
subplot.set_title("Tate Gallery Acquisitions", fontdict=title_font)
fig.show()


# Save to files
fig.savefig('plot.png')
fig.savefig('plot.svg', format='svg')





