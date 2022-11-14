# autofuse
This repository is the sample code for the paper
> Automatic Fusion Network for Cold-start CVR Prediction with Explicit Multi-Level Representation

### Environment
The code has been tested running under Python 3.6.10 and Centos7, with the following packages installed (along with their dependencies):
- tensorflow == 2.1.0
- numpy == 1.19.5
- pandas == 1.1.5
- keras == 2.9.0
- scikit-learn == 0.24.2
- scipy == 1.4.1

### Dataset
**Dataset Link:** https://nijianmo.github.io/amazon/index.html (Or https://jmcauley.uscd.edu/dataset/amazon)

merge the reviews information and meta information similar as https://github.com/zhougr1993/DeepInterestNetwork 

**Feature Meaning:**

- asin: ID of the product
- title: name of the product
- description: description of the product
- price: price in US dollars
- reviewerID: ID of the reviewer
- reviewTime: time of the review
- reviewerName: name of the reviewer
- vote: helpful votes of the review
- style: a disctionary of the product metadata
- reviewText: text of the review
- unixReviewTime: time of the review
- imageURL: url of the high resolution product image
- brand: brand name
- overall: rating of the product

Two special feaures
- popularity: historically cumulative conversions of each product in the dataset (statistical results)
- label: we set “overall” over 3 as conversion behavior, labeled 1, otherwise 0.

**Feature Classification:**

- user-side feautres: reviewerID,  reviewerName
- item-side features: asin, title, description, price, reviewTime, vote, style, reviewText, unixReviewTime, imageURL, brand
- dense features: price
- sparse features: reviewerID, reviewerName, asin, title, description,reviewTime, vote, style, reviewText, unixReviewTime, imageURL, brand
- coarse features(only item-side): price, vote, unixReviewTime, brand, style
- fine featuress(only item-side): asin, reviewText, title, summary, imageURL, description

### Running the code
The sample code is built based on Weichen Shen. (2017). DeepCTR: Easy-to-use, Modular and Extendible package of deep-learning based CTR models. https://github.com/shenweichen/deepctr.

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
$ python3 main.py
```
