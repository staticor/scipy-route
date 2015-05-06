# coding: utf-8


import numpy as np 
import pylab as pl
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
cos, sin = np.cos(x), np.sin(x)


#print x
 
# Matplotlib comes with  Matplotlib comes with a set of default settings that allow customizing all kinds of properties. You can control the defaults of almost every property in matplotlib: figure size and dpi, line width, color and style, axes, axis and grid properties, text and font properties and so on.


# In the script below, we’ve instantiated (and commented) all the figure settings that influence the appearance of the plot.
pl.figure(figsize=(8, 6), dpi=80)
pl.subplot(1, 1, 1)
pl.plot(x, cos)
pl.plot(x, sin)
pl.show()

