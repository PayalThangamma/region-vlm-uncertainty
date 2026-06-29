# Script 01 — Create Raw ROHE Dataset

## Objective
Plan is to generate ROHE-style dataset from the MS COCO validation set by selecting suitable object instances and creating the metadata and segmentation masks required for object removal.


## Why is this script necessary?
The Epistemic Uncertainty paper assumes that images already exist where an object has been removed. MS COCO does **not** contain these images. Therefore we must first build our own dataset. This script produces the raw dataset before inpainting.

## Inputs
```
data/
    coco/
        val2017/
        annotations/
            instances_val2017.json
```

## Outputs
```
data/
    rohe_raw/
        sample_000001/
            original.jpg
            mask.png
            mask_overlay.png
            metadata.json
```
and

```
manifest.json
```

### Step 1
Load COCO annotations.

### Step 2
Choose only target object categories.

Example

```
dog
cat
car
chair
...
```

### Step 3
Find every image containing those categories.

### Step 4
For every image, retrieve all object instances

### Step 5
Choose the largest instance.
Reason: Small objects produce unreliable CLIP token regions.

### Step 6
Compute
```
object_fraction =
object_area / image_area
```
Keep only 5% to 30%
Reason: Very small objects(almost no removed tokens), Very large objects(remove too much of the scene)

### Step 7
Create binary mask using annToMask()

### Step 8
Copy original image.

### Step 9
Create visualization mask_overlay for debugging.

### Step 10
Save metadata.
Reason:Later scripts should never need to query COCO again. Everything required is stored locally.

Example
```
target_object
question
ground_truth
annotation_id
source image
```

## Verification
Every sample should contain
```
original.jpg
mask.png
mask_overlay.png
metadata.json
```

## Failure cases
Missing image then skip sample
Missing annotation then skip sample
Bad segmentation then removed during quality filtering


# Script 02 — Prepare LaMa Input

## Objective
Convert the raw ROHE dataset into the input format required by **LaMa (Large Mask Inpainting)**.
After Script 01, each sample contains:
```text
sample_000001/
    original.jpg
    mask.png
```
LaMa expects images and masks in a single folder with matching filenames.
Therefore, we reorganize the dataset without modifying the images.

## Inputs
```text
data/
    rohe_raw/
        sample_*/
            original.jpg
            mask.png
```

## Outputs
```text
lama_input/
    sample_000001.png
    sample_000001_mask.png
    sample_000002.png
    sample_000002_mask.png
    ...
```

### Step 1
Create the output directory.
If it already exists, delete previous files to avoid mixing old and new runs.

### Step 2
Iterate over every sample folders

### Step 3
Verify that both files exist:
```text
original.jpg
mask.png
```
If either file is missing: Skip the sample and Print a warning.

### Step 4
Copy the files into `lama_input`.
Rename them as:
```text
sample_000001.png
sample_000001_mask.png
```

### Step 5
Count the successfully prepared samples.

## Why copy instead of moving?
We intentionally **copy** instead of moving because:

* `rohe_raw` remains an untouched source dataset.
* LaMa can be rerun multiple times.
* If LaMa fails, the original data is still available.
* Multiple inpainting methods could be tested later without rebuilding the dataset.

This separation makes the pipeline reproducible and safer.