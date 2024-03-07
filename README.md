# TO APPEAR IN SIGMOD 2024 : https://arxiv.org/abs/2402.17926

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
