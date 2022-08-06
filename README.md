# Reproduce-graduate-thesis
Repo for reproducing the results of my master's thesis

In this repo, the code can be found to reproduce most - if all - results from my graduate thesis. Do not expect to find clean code or proper comments, the work was performed properly but rushed. 

Setting up the datasets and experiments is quite a hassle- I will keep my datasets and files on my server for a while after graduation, as well as an `rsync` of my hpc home dir. Feel free to contact, and I'll send them over. 

## Preparation: Downloading datasets 
1. Download the ENA dataset and channel islands datasets from https://lila.science/datasets/channel-islands-camera-traps/ and https://lila.science/datasets/ena24detection. I suggest using `azcopy` as the datasets are quite large (89GB total). Take note of the paths to the images as well as the paths to the metadata (json) files. 
2. Install the dependencies in requirements.txt - this will take a while. 
3. Run the dataprep python files found in datasetprep/datasethandler/*.py to generate the pickle files for the data subsets. The dataprep files contain some general EDA used for the thesis report. Be sure to edit the `class config` classes for both of the files manually - and set the save_pickles flag to `True`. 
4. Save the generated pickle files somewhere- these are used for generating the various subsets


## Generating subsets
Generating the subsets is done by taking the pickle files from the last step, adding some flags in a config, then generating the required subset. 
The pipeline for generating the subsets is found in `generating_subsets\datapipeline.py`. The configs used are found in `generating_subsets\configs\*.yaml`. For example: 
```yaml
remove_existing: True 
sequential_data : False
pickle_path : "C:\\Projects\\seq2bbox\\data\\pickle\\ENA\\Test.pk"
dataset_path : "C:\\temp\\ENA_full\\"
image_path : "C:\\temp\\data_final\\ENA\\images\\ENA640xCropGNTest\\"
histlbp_path : "C:\\temp\\data_final\\ENA\\histlbp\\ENA640xCropGNTest\\"
label_path : "C:\\temp\\data_final\\ENA\\labels\\ENA640xCropGNTest\\"
generate_histlp : False 
generate_labels : True
convert_grayscale : True 
wavelet_compress : False
naive_compress : True 
resize : True 
image_size : 640 
# multiprocessing
chunksize : 8
max_workers : 8
```
Make sure you've activated your environment. 
Generate a config, run the datapileine with `path_to_py3_executable generating_subsets\datapipeline.py -c path_to_config`. 

Be careful with `remove_existing` - this removes all files from the paths submitted. 

## Running models and testing
### VAE
The VAE was modified from: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
For the VAE, you only need to update the `config` class to match your file locations and run the models. Each epoch, a checkpoint of the model is saved as a `.pt` file. 
### Ghostnet
Ghostnet was modified from: https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnet_pytorch

The dual ghostnet can be run with the `dual_main.py` file in `ghostnet\nets\`. 

The dataloader of ghostnet currently does not support a test set- just replace the validation set with a test set and run the validation to get test results. In addition, the dataloaders currently only support up to 10 classes 




## References: 

Letterbox: https://github.com/ultralytics/yolov5/blob/4d157f578a7bbff08d1e17a4e6e47aece4d91207/utils/augmentations.py#L91

Local Binary Patterns (mahotas): https://mahotas.readthedocs.io/en/latest/index.html


Part of background substraction logic: https://stackoverflow.com/questions/60646384/python-opencv-background-subtraction-and-bounding-box

VAE: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

seaborn plots: https://www.statology.org/seaborn-barplot-show-values/


TODO: createlabels (bbox label) coco to yolo...
TODO cv2