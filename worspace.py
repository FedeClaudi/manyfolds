import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 100)
y = np.sin(2.5 * np.pi * (x - 0.1))

y1 = np.sin(2 * np.pi * (x))


plt.plot(x, -y)
plt.plot(x, -y1, lw=3)

plt.plot([0, 1], [0, 0], color="black")
plt.plot([0, 0], [0, 1], color="black")

plt.show()
