#  Model summary

Here, two deep neural networks for primary grading of Korean commercial pig are proposed. 
1)  Back-fat thickness estimation network (BTENet), which simultaneously performs the back-fat area segmentation and thickness estimation.
2) Sex classification network (SCNet), which determines the sex classes of pig carcass.

# Implementation

## 1. Preparing
Our models were implemented by tensorflow 2.3 in Python 3.8.6. 
Pre-trained weights for BTENet and SCNet can be downloaded from  https://drive.google.com/drive/folders/1PRBpfRVALwiPbA6JYSB9V-jDl2Rr8FIJ?usp=sharing.
**important**
Both pre-trained weights must be placed at [CODE PATH]/weights for execution.
## 2. BTENet
### 2.1. Execution
```
python main_btenet.py --img [IMG] --out [OUT] --device [device]
```
1. [IMG]: image file path, which contain the head-side image of VCS-2000.
2. [OUT]: Path for saving the results.
3. [device]: GPU device number to use (default: 0).

**example**
```
python main_btenet.py --img data/head_img --out btenet_results --device 0
```
### 2.2 Output
1. bf.sol: A text file with a header line, and the one line per sample with 2 columnes. The first column is file name and another is predicted back-fat thickness (mm).
2. mask: A file path, which includes the predicted back-fat area mask.
3. paint: A file path, which includes visualized prediction results.
![bf](https://user-images.githubusercontent.com/71325306/147445925-c1e89b53-eb7e-4ae5-93de-574fe641ad8e.png)



## 3. SCNet
### 2.1. Execution
```
python main_scnet.py --img [IMG] --out [OUT] --device [device]
```
1. [IMG]: image file path, which contain the hip-side image of VCS-2000.
2. [OUT]: Path for saving the results.
3. [device]: GPU device number to use (default: 0).

**example**
```
python main_scnet.py --img data/hip_img --out scnet_results --device 0
```
### 2.2 Output
1. pred.sol: A text file with a header line, and the one line per sample with 3 columnes. Each columns mean file name, predicted sex class, and class probability
2. paint: A file path, which includes visualized prediction results.
![sex](https://user-images.githubusercontent.com/71325306/147446032-e604691d-dac9-4ed7-b0f6-6e9ae8688306.png)
