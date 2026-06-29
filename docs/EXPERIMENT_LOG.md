Date: 2026-06-29

Script: 01_create_rohe_dataset.py
Goal:Create a raw ROHE-style dataset from COCO by selecting medium-sized object instances and generating segmentation masks.
Result:Successfully generated 636 candidate samples.
Output:
data/rohe_raw/
    sample_000001/
    sample_000002/
    ...
    sample_000636/
Files created per sample:
- original.jpg
- mask.png
- mask_overlay.png
- metadata.json
Only candidate samples were generated. Object removal (LaMa) is performed in the next stage.

Script: 02_prepare_lama_input.py
Objective:Convert the raw ROHE dataset into the input format required by LaMa.
Input:  
- `data/rohe_raw/`
Output:  
- `lama_input/`
Result:Successfully prepared 636 samples for LaMa.
Verified:636 original images and 636 corresponding masks were copied into `lama_input`.