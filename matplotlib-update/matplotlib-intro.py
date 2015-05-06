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
pl.plot(x, cos, color='b',linestyle='--', linewidth=1, label='cosine')
pl.plot(x, sin, color='r',linestyle='-', linewidth=2.5, label='sine')
pl.xlim(-4.1, 4.1)
pl.ylim(-1.1, 1.1)
pl.xticks(np.linspace(-4.1, 4.1, 9), list('abcdefghi'))
pl.xticks(np.linspace(-4.1, 4.1, 9))
pl.yticks(np.linspace(-1.1, 1.1, 5))

# axis 控制 spines.
ax = pl.gca()  # get current axis
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))


# add legend
pl.legend(loc='upper left')

# annotate
t = 2 * np.pi / 3 
pl.plot([t, t], [0, np.cos(t)], 'b--')
pl.scatter([t, ], [np.cos(t), ], s=50, color='b')

pl.annotate(r'$sin(\frac{2\pi}{3}) = \frac{\sqrt{3}}{2}$',
            xy=(t, np.sin(t)), xycoords='data',
            xytext=(+10, +30), textcoords='offset points',
            fontsize=16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2")
            )

pl.plot([t,t], [0, np.sin(t)], 'r--')
pl.scatter([t, ], [np.sin(t),], s=60, color='r')


#savefig('exercise01.jpg')
pl.show()




