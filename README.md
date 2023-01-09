# Multi-omic Analysis of Lung Cancer using Autoencoders

## 02-718 Computational Medicine Course Project

## Contributors
- **Arnav Gupta**  
- **Ketaki Ghatole** 


## Description
Lung Cancer is one of the most common and fatal cancer types. It accounts for 11.4% of total cancer cases leading to over one million deaths annuall. The 
treatment greatly depends on accurate tumor classification. Histologically, there are two main subtypes of Lung Cancer: 
Adenocarcinoma (ALC) and Squamous Cell Lung Carcinoma (SCLC). The high rate of incidence and mortality of lung cancer makes its early detection and
accurate sub-type classification crucial in the course of treatment

This project used publicly available gene expression and CNV data and clinical information from The Cancer Genome Atlas (TCGA).

## Workflow
<img width="811" alt="Screenshot 2023-01-09 at 1 11 58 PM" src="https://user-images.githubusercontent.com/52592007/211315953-5859396e-ed52-4c4c-b9bd-d61b6010badd.png">

1. This project aims to use Autoencoders for unsupervised for dimensionality reduction. We built 4 different autoencoders using pytorch - GE, CNV, GE + CNV, GE + CNV ensemble  
2. We visualised the data using PCA and t-SNE
3. We further aim to classify between the ALC and SCLC using supervised machine learning models Support Vector Machine (SVM) and Random Forest and compare the results for mono-omic and multi-omic data using a 5-fold Cross
Validation

## Results
<img width="769" alt="Screenshot 2023-01-09 at 1 02 47 PM" src="https://user-images.githubusercontent.com/52592007/211314344-e1f6e17e-fbc7-4139-8e24-3eb2ce0761b9.png">

1. CNV + GE Ensemble performed slightly better than the mono-omic data
2. The results from the GE + CNV combined model were better than the CNV but slightly lower than the GE models
3. We believe these results are promising and more work is required in improving the unsupervised feature selection using autoencoders by optimizing the parameters 
