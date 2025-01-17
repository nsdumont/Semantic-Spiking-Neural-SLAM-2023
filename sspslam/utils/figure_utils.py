import sys, os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the correct matplotlibrc
# mpl.rc_file(os.path.join(os.path.dirname(__file__), 'matplotlibrc'))
# mpl.rcParams["text.usetex"] = True
# mpl.rcParams["text.latex.preamble"] = r"""
#     \renewcommand*\familydefault{\sfdefault}
#     \renewcommand{\vec}[1]{\mathbf{#1}}
#     \newcommand{\mat}[1]{\mathbf{#1}}
# """

# Load the correct matplotlibrc
mpl.rc_file(os.path.join(os.path.dirname(__file__), 'matplotlibrc'))
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"""
    \usepackage{siunitx}
    \usepackage{libertine}
    \usepackage{libertinust1math}
    \usepackage{mathrsfs}
    \usepackage{amssymb}
    \renewcommand*\familydefault{\sfdefault}
    \renewcommand{\vec}[1]{\mathbf{#1}}
    \newcommand{\mat}[1]{\mathbf{#1}}
"""

blues = ["#729fcfff", "#3465a4ff", "#204a87ff"][::-1]
reds = ["#ef2929ff", "#cc0000ff", "#a40000ff"][::-1]
greens = ["#8ae234ff", "#73d216ff", "#4e9a06ff"][::-1]
oranges = ["#fcaf3eff", "#f57900ff", "#ce5c00ff"][::-1]
purples = ["#ad7fa8ff", "#75507bff", "#5c3566ff"][::-1]
yellows = ["#fce94fff", "#edd400ff", "#c4a000ff"][::-1]
grays = [
    "#eeeeecff", "#d3d7cfff", "#babdb6ff", "#888a85ff", "#555753ff",
    "#2e3436ff"
][::-1]



def circles(x, y, s, c='b', vmin=None, vmax=None, ax=None, **kwargs):
    """
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """

    import numpy as np
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    if ax is None:    
        ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection


def save(fig, filename, suffix=""):
    # Special treatment for PDFs. We need to run the resulting PDF
    # through Ghostscript to
    # a) trim the figures properly
    # b) subset fonts
    target_file, target_ext = os.path.splitext(filename)
    target_file += suffix
    if target_ext == ".pdf":
        target = target_large = target_file + ".large" + target_ext
    else:
        target = target_file + target_ext
    print("Saving to", target)
    fig.savefig(target,
                bbox_inches='tight',
                pad_inches=0.05,
                transparent=True)

    if target_ext == ".pdf":
        import subprocess
        import re

        # Extract the bounding box
        print("Extracting bounding box of file", target_large)
        gs_out = subprocess.check_output(
            ["gs", "-o", "-", "-sDEVICE=bbox", target_large],
            stderr=subprocess.STDOUT)
        pattern = r"^%%HiResBoundingBox:\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
        x0, y0, x1, y1 = map(
            float,
            re.search(pattern, str(gs_out, "utf-8"),
                      re.MULTILINE).groups())

        # Add a small extension to the bounding box
        pad = 0.5
        x0 -= pad
        x1 += pad
        y0 -= pad
        y1 += pad

        # Run ghostscript again to crop the file
        # See https://stackoverflow.com/a/46058965
        target = target_file + target_ext
        print("Optimising PDF and saving to", target)
        subprocess.check_output([
            "gs", "-o", target, "-dEmbedAllFonts=true",
            "-dSubsetFonts=true", "-dCompressFonts=true",
            "-dPDFSETTINGS=/prepress", '-dDoThumbnails=false',
            "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
            f"-dDEVICEWIDTHPOINTS={x1 - x0}",
            f"-dDEVICEHEIGHTPOINTS={y1 - y0}", "-dFIXEDMEDIA", "-c",
            f"<</PageOffset [-{x0} -{y0}]>>setpagedevice", "-f",
            target_large
        ])

        # Remove the large temporary file
        os.unlink(target_large)

def outside_ticks(ax):
    ax.tick_params(direction="out", which="both")

def remove_frame(ax):
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)

def add_frame(self, ax):
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(True)

def annotate(ax,
             x0,
             y0,
             x1,
             y1,
             s=None,
             ha="center",
             va="center",
             fontdict=None,
             zorder=None,
             color="k"):
    ax.plot([x0, x1], [y0, y1],
            color=color,
            linewidth=0.5,
            linestyle=(0, (1, 1)),
            clip_on=False,
            zorder=zorder)
    ax.plot([x0], [y0],
            'o',
            color=color,
            markersize=1,
            clip_on=False,
            zorder=zorder)
    if not s is None:
        ax.text(x1,
                y1,
                s,
                ha=ha,
                va=va,
                bbox={
                    "pad": 1.0,
                    "color": "w",
                    "linewidth": 0.0,
                },
                fontdict=fontdict,
                zorder=zorder)


def vslice(ax, x, y0, y1, **kwargs):
    ax.plot([x, x], [y0, y1],
            'k-',
            linewidth=0.75,
            clip_on=False,
            **kwargs)
    ax.plot(x, y0, 'k_', linewidth=0.75, clip_on=False, **kwargs)
    ax.plot(x, y1, 'k_', linewidth=0.75, clip_on=False, **kwargs)


def hslice(ax, x0, x1, y, **kwargs):
    ax.plot([x0, x1], [y, y],
            'k-',
            linewidth=0.75,
            clip_on=False,
            **kwargs)
    ax.plot(x0, y, 'k|', linewidth=0.75, clip_on=False, **kwargs)
    ax.plot(x1, y, 'k|', linewidth=0.75, clip_on=False, **kwargs)


def timeslice(ax, x0, x1, y, **kwargs):
    hslice(ax, x0, x1, y, **kwargs)

