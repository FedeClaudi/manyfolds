import numpy as np
import matplotlib.pyplot as plt



S = np.linspace(0, .999, 100)

x = np.cos(2 * np.pi * S)
y = np.sin(2 * np.pi * S)
z = np.sin(4 * np.pi * S)


fig = plt.figure()
ax = plt.axes(projection='3d')


ax.plot3D(x, y, z)
plt.show()
