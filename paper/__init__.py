from pathlib import Path

images_folder = Path(
    "/Users/federicoclaudi/Documents/Github/manyfolds/paper/images"
)

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
