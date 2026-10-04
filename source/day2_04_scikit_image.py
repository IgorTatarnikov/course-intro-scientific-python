# %% [markdown]
# # Day 2 · scikit-image: counting nuclei
#
# **Block:** scikit-image (45 min, including slides and live coding)
#
# **Task:** count the nuclei in a fluorescence microscopy image of human cells, and
# measure their sizes. You will:
#
# 1. **Look** at the image as a NumPy array (3 min)
# 2. **Filter** it to reduce noise (4 min)
# 3. **Threshold** it into foreground and background (8 min)
# 4. **Clean up and label** the objects, then count them (8 min)
# 5. **Measure** each object with `regionprops` (7 min)
#
# Sections 1 to 3 are the first "Your turn" slide, sections 4 and 5 the second.
# 6. **Stretch:** separate touching nuclei with a watershed, then try the SPL coins
#    exercises
#
# Our recording has no images, so this block switches to microscopy, the other
# place most of you will meet image data. The steps (filter, threshold, label,
# measure) are the same for cells in a two-photon recording or in brain
# sections.
#
# The image is `skimage.data.human_mitosis()`: human cells in culture,
# some of them dividing, stained for DNA. It comes from Moffat *et al.*, *Cell*
# 124:1283 (2006), courtesy of David Root, licensed CC0.
#
# Further material: [SPL, scikit-image: image
# processing](https://lectures.scientific-python.org/packages/scikit-image/index.html)
# and [Image manipulation and processing using NumPy and
# SciPy](https://lectures.scientific-python.org/advanced/image_processing/index.html)
# (CC BY 4.0), plus the [BioImage Analysis
# Notebooks](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import skimage as ski

# %% [markdown]
# ## 1. Look at the image
#
# Load `../data/human_mitosis.png` with `ski.io.imread`, then answer:
#
# * What are its shape, dtype, minimum and maximum?
# * Display it with `imshow` and a grey colormap. Then plot a histogram of the pixel
#   values (`ax.hist(image.ravel(), bins=...)`). What do the two humps tell you?

# %%
image = ski.io.imread("../data/human_mitosis.png")

# %% tags=["solution"]
print(image.shape, image.dtype, image.min(), image.max())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(image, cmap="gray")
axes[0].set_axis_off()
axes[1].hist(image.ravel(), bins=128, log=True)
axes[1].set_xlabel("pixel value")
axes[1].set_ylabel("count (log scale)");

# %% [markdown] tags=["answer"]
# A 512 x 512 `uint8` image (values 0 to 255). Most pixels are dark background,
# which makes the big hump at low values. The nuclei make a long tail of bright
# values. The log scale on the counts makes the tail visible.

# %% [markdown]
# ## 2. Filter
#
# Thresholding single pixels is sensitive to noise. Smooth the image with
# `ski.filters.gaussian` using `sigma=1`, and call the result `smoothed`. Show
# the image and `smoothed` side by side, zoomed in on a corner (for example
# `[:100, :100]`).
#
# What is the dtype and range of `smoothed`? Why did it change?

# %% tags=["solution"]
smoothed = ski.filters.gaussian(image, sigma=1)
print(smoothed.dtype, smoothed.min(), smoothed.max())

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image[:100, :100], cmap="gray")
axes[0].set_title("original")
axes[1].imshow(smoothed[:100, :100], cmap="gray")
axes[1].set_title("gaussian, sigma=1")
for ax in axes:
    ax.set_axis_off()

# %% [markdown] tags=["answer"]
# `smoothed` is `float64` between 0 and 1. Most scikit-image filters convert
# integer images to floats in [0, 1] (`ski.util.img_as_float`), so the averaging
# cannot overflow or round. Keep this in mind when you compare thresholds between
# the two.

# %% [markdown]
# ## 3. Threshold
#
# 1. Compute an automatic threshold for `smoothed` with `ski.filters.threshold_otsu`.
# 2. Make the boolean image `binary = smoothed > threshold` and display it.
# 3. Try `ski.filters.try_all_threshold(smoothed)`. Which methods look better or
#    worse than Otsu here?

# %% tags=["solution"]
threshold = ski.filters.threshold_otsu(smoothed)
binary = smoothed > threshold
print("Otsu threshold:", threshold)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(binary, cmap="gray")
ax.set_axis_off()

# %% tags=["solution"]
fig, ax = ski.filters.try_all_threshold(smoothed, figsize=(8, 10), verbose=False)

# %% [markdown] tags=["answer"]
# Isodata, Minimum and Otsu give almost the same result here. Li, Mean and
# Triangle pick lower thresholds, so nuclei come out fatter and more of them
# merge with their neighbours. Yen picks a threshold far too high and keeps only
# a few bright specks. No method is best for every image, so always look at the
# result.

# %% [markdown]
# ## 4. Clean up, label and count
#
# 1. Remove specks smaller than about 30 pixels with
#    `ski.morphology.remove_small_objects(binary, max_size=30)`.
# 2. Remove nuclei cut off by the image edge with `ski.segmentation.clear_border`,
#    because they would distort the size measurements. Call the result `cleaned`.
# 3. Give each connected object its own integer with `ski.measure.label`. How many
#    nuclei are there?
# 4. Show the labels on top of the image with `ski.color.label2rgb(labels,
#    image=image, bg_label=0)`.
#
# Note: older tutorials use `remove_small_objects(..., min_size=30)`. In
# scikit-image 0.26 that argument is deprecated in favour of `max_size`, which
# removes objects smaller than **or equal to** the value.

# %% tags=["solution"]
cleaned = ski.morphology.remove_small_objects(binary, max_size=30)
cleaned = ski.segmentation.clear_border(cleaned)
labels = ski.measure.label(cleaned)
print("Number of nuclei:", labels.max())

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(ski.color.label2rgb(labels, image=image, bg_label=0))
ax.set_axis_off()

# %% [markdown]
# ## 5. Measure
#
# `ski.measure.regionprops_table` measures every labelled object and returns a
# dictionary that pandas can turn straight into a DataFrame.
#
# 1. Measure `label`, `area`, `eccentricity` and `intensity_mean` (pass
#    `intensity_image=image` for the last one) into a DataFrame `nuclei`.
# 2. Plot a histogram of the areas. Is it one population, or is there a tail?
# 3. Look at the largest objects: are they big nuclei, or something else?
#    (Hint: `nuclei.nlargest(5, "area")`, then find those labels in the image.)

# %% tags=["solution"]
nuclei = pd.DataFrame(
    ski.measure.regionprops_table(
        labels,
        intensity_image=image,
        properties=["label", "area", "eccentricity", "intensity_mean"],
    )
)
nuclei.describe().round(2)

# %% tags=["solution"]
fig, ax = plt.subplots()
ax.hist(nuclei["area"], bins=40)
ax.set_xlabel("area (pixels)")
ax.set_ylabel("number of objects");

# %% tags=["solution"]
largest = nuclei.nlargest(5, "area")
print(largest)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(image, cmap="gray")
ax.contour(np.isin(labels, largest["label"]), levels=[0.5], colors="red")
ax.set_axis_off()

# %% [markdown] tags=["answer"]
# Most objects are around 80 to 130 pixels, with a long tail of large ones. The
# largest "nuclei" are mostly two or more touching nuclei that thresholding
# merged into one object, so they inflate the sizes and deflate the count. The
# watershed in the stretch section separates them.

# %% [markdown]
# ## 6. Stretch
#
# ### Separate touching nuclei with a watershed
#
# Touching nuclei form a single connected blob. The classic fix:
#
# 1. Compute the distance from each foreground pixel to the background:
#    `sp.ndimage.distance_transform_edt(cleaned)`. Nucleus centres are maxima.
# 2. Find those maxima with `ski.feature.peak_local_max(distance, min_distance=7,
#    labels=labels)`. It returns coordinates.
# 3. Turn them into a marker image (0 everywhere, 1, 2, 3, ... at the maxima).
# 4. Flood `-distance` from the markers with `ski.segmentation.watershed(-distance,
#    markers, mask=cleaned)`.
#
# How many nuclei do you count now? Compare `label2rgb` images before and after.

# %% tags=["solution"]
distance = sp.ndimage.distance_transform_edt(cleaned)
coords = ski.feature.peak_local_max(distance, min_distance=7, labels=labels)
markers = np.zeros_like(labels)
markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
separated = ski.segmentation.watershed(-distance, markers, mask=cleaned)
print("Before:", labels.max(), "after watershed:", separated.max())

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, lab, title in zip(axes, [labels, separated], ["connected components", "watershed"]):
    ax.imshow(ski.color.label2rgb(lab[150:350, 150:350], image=image[150:350, 150:350], bg_label=0))
    ax.set_title(title)
    ax.set_axis_off()

# %% [markdown] tags=["solution"]
# `min_distance` controls how eagerly blobs are split. Too small, and elongated
# nuclei get cut in two; too large, and touching pairs stay merged. Check the
# result by eye on a zoomed-in region, as above.
#
# ### Save the labels for napari
#
# The napari notebook at the end of the day recomputes this segmentation itself,
# so you do not need to save anything. If you want to look at *your* segmentation
# there, save it with `np.save("my_labels.npy", separated)`.

# %% [markdown]
# ### SPL exercises on the coins image
#
# From [SPL, scikit-image](https://lectures.scientific-python.org/packages/scikit-image/index.html):
#
# * **Segmentation:** load `ski.data.coins()`. Separate the coins from the background
#   by trying several methods: Otsu thresholding, adaptive thresholding
#   (`ski.filters.threshold_local`), and watershed or random walker segmentation.
#   Post-process if needed.
# * **Measurement:** from your binary coins image, label the coins and compute the
#   size and eccentricity of each.
# * **Colour:** open a colour image (for example `ski.data.astronaut()`), plot the
#   histogram of each colour channel, then convert it to grey with
#   `ski.color.rgb2gray` and plot that histogram.
