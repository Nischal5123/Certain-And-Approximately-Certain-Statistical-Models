# SIGMOD 2024

## 1. Synthetic Dataset

The source code for three models (Linear Regression, Linear SVM, and Kernel SVM) can be found within the "Synthetic" folder.

## 2. ActiveClean Benchmark with scikit-learn

In some versions of scikit-learn, when running the benchmark method ActiveClean (code from the authors [here](https://www.dropbox.com/sh/r2vv252m5lnqpmm/AAAMj0WRaZX9EKH_8dLOHQpIa?dl=0&preview=activeclean_sklearn.py)), you may encounter an error related to the 'loss' parameter of SGDClassifier. If you face this issue, please replace all instances of "log" with "log_loss" in the SGDClassifier within the ActiveClean functions.

-----------------------------------------------------------------------------------------

### New Revision Experiment Details

## 3. Real-World Dataset-with-Random-Corruption

This section includes three UCI datasets that were utilized in our paper. Each Dataset is divided into its own folder.
- If you encounter the same error as described in point 2, kindly apply the modifications to the ActiveClean functions.
- Data files are available through the links below to the UCI database:
  1. [Malware](https://doi.org/10.24432/C5HG8D)
  2. [Gisette](https://doi.org/10.24432/C5HP5B)
  3. [Tuandromd](https://doi.org/10.24432/C5560H)
 
For Running Experiments for instance for TUANDROMD. python3 TUANDROMD-DRIVER.py 


## 4. Real-World-Dataset

This section includes 8 real-world datasets with inherent missing values.
- Every dataset has its specific file for running the code.
- Some large datasets example Intel_Sensor can be found in their original source, all are cited in the paper.
- Deep Learning based imputation (MIWAE) is implemented from the following repository: [https://github.com/vanderschaarlab/hyperimpute](https://github.com/vanderschaarlab/hyperimpute)
- KNNImputer is implemented based on sklearn: [https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
- MeanImputer is implemented based on sklearn: [https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

-For Running Experiments for instance for water potability. python3 Water_Portability.py.


## 5. Approximate-Certain-Model

This section includes code for ACM experiments.
- Every dataset has its specific file for running the code.
- Make sure the filepath is pointed to the correct directory.
-For Running Experiments for instance for breast cancer. python3 breast_cancer_ACM.py.

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{10.1145/3654929,
author = {Zhen, Cheng and Aryal, Nischal and Termehchy, Arash and Chabada, Amandeep Singh},
title = {Certain and Approximately Certain Models for Statistical Learning},
year = {2024},
issue_date = {June 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {2},
number = {3},
url = {https://doi.org/10.1145/3654929},
doi = {10.1145/3654929},
abstract = {Real-world data is often incomplete and contains missing values. To train accurate models over real-world datasets, users need to spend a substantial amount of time and resources imputing and finding proper values for missing data items. In this paper, we demonstrate that it is possible to learn accurate models directly from data with missing values for certain training data and target models. We propose a unified approach for checking the necessity of data imputation to learn accurate models across various widely-used machine learning paradigms. We build efficient algorithms with theoretical guarantees to check this necessity and return accurate models in cases where imputation is unnecessary. Our extensive experiments indicate that our proposed algorithms significantly reduce the amount of time and effort needed for data imputation without imposing considerable computational overhead.},
journal = {Proc. ACM Manag. Data},
month = may,
articleno = {126},
numpages = {25},
keywords = {data preparation, data quality, uncertainty quantification}
}
