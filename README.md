# autofuse
This repository is the sample code for the paper
> Towards Explicitly Learning Multi-Level Representations for Cold-start Advertisement

### Environment
The code has been tested running under Python 3.6.10 and Centos7, with the following packages installed (along with their dependencies):
- tensorflow == 2.1.0
- numpy == 1.19.5
- pandas == 1.1.5
- keras == 2.9.0
- scikit-learn == 0.24.2
- scipy == 1.4.1

### Dataset
Dataset Link: https://nijianmo.github.io/amazon/index.html (Or https://jmcauley.uscd.edu/dataset/amazon)
Feature meaning:
- asin: ID of the product
- title:
- 
Dataset Preprocess:
Step1 : merge the reviews information and meta information similar as https://github.com/zhougr1993/DeepInterestNetwork 
Step2: 

### Running the code
Step1: 
```
$ git clone https://github.com/shenweichen/DeepCTR.git 
```

Step1: 
```
$ mv autofuse.py DeepCTR/deepctr/models
```

Step3:
```
$ python main.py
```
