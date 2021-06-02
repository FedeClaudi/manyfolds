import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

f, ax = plt.subplots()
ax.set(
    title=r"$I f^1 f^2 x^1 x^2 x^{-1} x(p) x^{-1} \circ f^1 x^{-1} \circ f^2 p U \phi \mathcal{M} \phi(p) \phi \circ x^{-1} \circ f^{12} e_{12} 64 l N=123$"
)

f.savefig("letter.svg", format="svg")
