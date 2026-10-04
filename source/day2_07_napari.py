# %% [markdown]
# # Day 2 · Whirlwind tour: napari
#
# **Block:** whirlwind tour, napari (about 12 min: demo, then try it)
#
# **Problem it solves:** *looking* at N-D images and their annotations
# interactively: zooming, toggling layers, scrolling through z or time, and fixing
# segmentations by hand. napari is a desktop viewer you drive from Python, and
# everything you change in the window is a NumPy array you can read back.
#
# napari opens a **window on your computer**. It does not work on JupyterHub or a
# remote server. If nothing appears, look behind your other windows.
#
# Material follows the [napari tutorials](https://napari.org/stable/tutorials/),
# in particular [Getting started](https://napari.org/stable/tutorials/fundamentals/getting_started.html)
# and the [labels layer](https://napari.org/stable/howtos/layers/labels.html) guide.

# %% [markdown]
# ## The data: the segmentation from the scikit-image notebook
#
# These cells redo the scikit-image block's segmentation (including the watershed
# from its stretch section), so this notebook works whether or not you finished
# that one.

# %%
import numpy as np
import scipy as sp
import skimage as ski

image = ski.io.imread("../data/human_mitosis.png")
smoothed = ski.filters.gaussian(image, sigma=1)
binary = smoothed > ski.filters.threshold_otsu(smoothed)
cleaned = ski.segmentation.clear_border(ski.morphology.remove_small_objects(binary, max_size=30))
labels = ski.measure.label(cleaned)

distance = sp.ndimage.distance_transform_edt(cleaned)
coords = ski.feature.peak_local_max(distance, min_distance=7, labels=labels)
markers = np.zeros_like(labels)
markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
nuclei = ski.segmentation.watershed(-distance, markers, mask=cleaned)
print("nuclei:", nuclei.max())

# %% [markdown]
# ## Demo
#
# The cells below open napari. They are marked so the automatic course build skips
# them, but run them normally in Jupyter.
#
# `%gui qt` connects Jupyter to the Qt event loop that napari's window needs.

# %% tags=["no-execute"]
# %gui qt

# %% tags=["no-execute"]
import napari

viewer = napari.Viewer()
viewer.add_image(image, name="nuclei image", colormap="gray")
viewer.add_labels(nuclei, name="segmentation", opacity=0.4)
viewer.add_points(coords, name="centres", size=4, face_color="yellow")

# %% [markdown]
# Things to show in the window:
#
# * Toggle layers with the eye icon, and change opacity and colormaps in the layer controls
# * Hover over a nucleus: the status bar shows its label number
# * Select the **segmentation** layer and use the **fill bucket** (shortcut
#   <kbd>4</kbd>) or the **brush** (<kbd>2</kbd>) to edit labels. Use <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>
#   + <kbd>Z</kbd> to undo.
# * A layer's `.data` is the live NumPy array. Edits in the window change it:

# %% tags=["no-execute"]
edited = viewer.layers["segmentation"].data
print("labels in the layer now:", len(np.unique(edited)) - 1)

# %% [markdown]
# ## Try it
#
# 1. Find a nucleus that the watershed split in two, or two nuclei it merged.
#    Fix it in the viewer: use the fill bucket to merge two labels into one, or
#    paint a new label (set the label number above the largest one) to split them.
# 2. Read the edited labels back into Python and recount the nuclei.
# 3. Add the **Otsu binary mask** as a second labels layer (`viewer.add_labels(binary.astype(int))`)
#    and flick between it and the watershed result. Where do they differ?
#
# Bonus: `viewer.screenshot("figure.png")` saves exactly what you see, and
# `viewer.add_image(...)` accepts 3-D and 4-D arrays, which get sliders for the
# extra dimensions.

# %% tags=["solution", "no-execute"]
# 2. Read the edits back and recount.
edited = viewer.layers["segmentation"].data
print("nuclei after editing:", len(np.unique(edited)) - 1)

# 3. Compare with the plain threshold.
viewer.add_labels(binary.astype(int), name="otsu binary", opacity=0.4)

# %% [markdown] tags=["answer"]
# Apart from the specks and the nuclei touching the image edge, which the cleaning
# step removed, the two layers agree on *which pixels* are nucleus. They differ in
# *how those pixels are split into objects*. The places worth checking by eye are
# clumps where the watershed drew a dividing line.
#
# **Where next:** the [napari tutorials](https://napari.org/stable/tutorials/) and
# the [napari hub](https://napari-hub.org) of plugins, many of them for bioimage
# analysis.
