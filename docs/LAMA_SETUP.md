### Why LaMa is needed?
to Generate object-removed images using inpainting while preserving the surrounding scene.

### Input
```
lama_input/
sample_000001.png
sample_000001_mask.png
...
```
### Output
```
lama_output/
sample_000001.png
...
```

### Environment
Download lama repository(folder named big-lama) from https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips 
Install anaconda or miniconda and open the command prompt

### Command
change path to directory to big-lama location
```
conda activate lama39
python bin\predict.py model.path=[path_to_downloaded_lama_repo] indir=[input_images_folder_path] outdir=[path_where_outputs_are_to_be_saved]
```

### Verification
Expected: 636 input images implies 636 inpainted images

### Common errors
```
no module named saicinpainting
```
Solution:
```
$env:PYTHONPATH = (Get-Location).Path
python -c "import saicinpainting; print(saicinpainting.__file__)"
```