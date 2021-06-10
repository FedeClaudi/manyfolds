import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update(
    {
        "font.size": 8,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}",
    }
)

f, ax = plt.subplots()
ax.set(title=r"$S^1 S^2 \mathbb{R}^1 \mathbb{R}^2 C T^2$")

f.savefig("letter.svg", format="svg")
