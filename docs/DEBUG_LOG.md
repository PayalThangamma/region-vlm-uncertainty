## Debug 001 — Permission Denied in `token_regions/` After HPC Upload

**Date:** 2026-06-30  
**Stage:** Stage 09 — Upload and Verify HPC Inputs  
**Location:** HPC cluster

### Problem

After uploading and extracting `outputs/hpc_inputs.zip` on the HPC cluster, the image folders were accessible and had the correct counts:

```text
original_images: 522
removed_images: 522
removed_images_jpg: 522
````

However, checking `token_regions/` produced many permission errors:

```text
find: ‘hpc_inputs/token_regions/sample_xxxxxx’: Permission denied
```

The token-region directory count initially returned:

```text
0
```

### Cause

The extracted `token_regions/` folders had incorrect permissions after transfer/extraction.

### Fix

The permissions were fixed using:

```bash
chmod -R u+rwX hpc_inputs
```

### Verification

After applying the permission fix, the folder and file counts were checked again.

Expected verified counts:

```text
token_regions: 522
token_to_region.json: 522
region_counts.json: 522
metadata.json: 522
```

### Status

Resolved.

The HPC input package is now accessible and ready for LLaVA baseline evaluation.

```

