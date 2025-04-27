# Titanic Classification and Regression (No Libraries)

This project implements **Logistic Regression** and **Linear Regression** from scratch (without using any libraries such as NumPy, Pandas, or Matplotlib) to perform:

- **Classification**: Predicting whether a passenger survived.
- **Regression**: Predicting the fare paid by a passenger.

## 🚀 Project Structure

- `data_loader.py`: Main script that:
  - Loads and processes the Titanic dataset.
  - Trains logistic regression for classification (`Survived`).
  - Trains linear regression for regression (`Fare`).
  - Evaluates both models.

## 🔍 Features Used

- Pclass (Passenger Class)
- Sex (Male/Female)
- Age (Missing values filled with mean ~29.7)
- SibSp (Number of siblings/spouses aboard)
- Parch (Number of parents/children aboard)

## 🧠 Models

### Classification - Logistic Regression
- Predicts whether the passenger survived (0 or 1)
- Evaluated using:
  - Accuracy
  - Confusion Matrix

### Regression - Linear Regression
- Predicts fare value
- Evaluated using:
  - Mean Absolute Error (MAE)

## 📂 Dataset

The dataset used is the classic **Titanic Dataset** (`train.csv`).

## 🛠️ Requirements

Nothing!  
This project is **pure Python**, no external libraries are used. Just run:

## Note: 
Libraries in this section are used for comparison only!
```bash
python data_loader.py
```
-----

# K-Means Clustering Project Without External Libraries

## Project Description
This project is an implementation of the **K-Means** clustering algorithm from scratch, without using any external libraries such as `scikit-learn` or `numpy`.  
The data used in this project comes from a `train.csv` file (e.g., Titanic dataset) with some preprocessing steps applied.

## Key Features
- Full implementation of the K-Means algorithm from scratch.
- Handling missing data by substituting default values.
- Manual computation of Euclidean distance.
- Calculation of **Sum of Squared Errors (SSE)** to evaluate clustering quality.

## How to Use
1. Ensure that the `train.csv` file is located in the same directory as the script.
2. Run the script:
   ```bash
   python your_script_name.py
   ```
3. The script will:
   - Load and read the data.
   - Preprocess the data and convert it into numerical format.
   - Apply the K-Means algorithm to cluster the data.
   - Print the number of elements in each cluster and the cluster centers (centroids).
   - Calculate and display the SSE value.

## Important Note
- Missing values in the `Age` column are replaced with 30.0, and missing values in the `Fare` column are replaced with 10.0.

## Requirements
- Python 3.x

## Project Files
- `main.py`: Contains the implementation of the K-Means algorithm.
- `data_loading.py`: Contains the `read_csv` function to load data from a CSV file.
- `train.csv`: The dataset file (must be provided in the same directory).

---

# Support Vector Machine HARD MARGINS(SVM) Implementation in Pure Python

This repository contains a pure Python implementation of a **hard-margin Support Vector Machine (SVM)** for binary classification. The implementation is designed to work with the Titanic dataset, where the goal is to predict whether a passenger survived (`1`) or did not survive (`0`).

The code is written entirely in Python without relying on external libraries like `numpy`, `scikit-learn`, or `pandas`. It demonstrates how to preprocess data, train an SVM model, and evaluate its performance using basic Python constructs.


---

## Overview

Support Vector Machines (SVMs) are powerful machine learning models used for classification and regression tasks. This implementation focuses on training a **hard-margin SVM** using only pure python. The model is applied to the Titanic dataset to predict survival based on features like age, fare, sex, and others.

Since this is a hard-margin SVM, it assumes the data is linearly separable. For real-world datasets, consider extending the implementation to support soft-margin SVMs or kernelized SVMs.

---

## Features

- **Pure Python**: No external libraries like `numpy` or `scikit-learn` are used.
- **Custom Quadratic Programming Solver**: Implements gradient descent to solve the QP problem.
- **One-Hot Encoding**: Converts categorical features into binary vectors for better representation.
- **Feature Scaling**: Standardizes numerical features to ensure equal contribution to the model.
- **Titanic Dataset**: Works with the Titanic dataset to predict survival outcomes.

---

## Dataset

The code uses the Titanic dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/c/titanic/data). The dataset contains information about passengers, including:
- **Features**: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex`, `Embarked`.
- **Target Variable**: `Survived` (1 = Survived, 0 = Did Not Survive).

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/svm-pure-python.git
   cd svm-pure-python
   ```

2. Ensure you have Python 3.x installed:
   ```bash
   python --version
   ```

3. Place the Titanic dataset (`train.csv`) in the appropriate directory:
   ```
   /path/to/project/data/train.csv
   ```

4. Run the script:
   ```bash
   python hardmargins.py
   ```

---

## Usage

### Step 1: Preprocess the Data
The `preprocess_data` function handles:
- Missing value imputation.
- One-hot encoding for categorical features (`Sex`, `Embarked`).
- Feature scaling for numerical features.

### Step 2: Train the Model
The `svm_train_hard_margin` function trains the SVM by solving the quadratic programming problem. It computes the weights ($w$) and bias ($b$) using the support vectors.

### Step 3: Evaluate the Model
The `svm_predict` function predicts the class labels for new data points. Accuracy is computed as the percentage of correct predictions.

---

## Code Structure

The project consists of two main files:

### 1. `data_loading.py`
Handles loading and parsing the Titanic dataset from a CSV file. It reads the raw data and returns the header and rows.

### 2. `hardmargins.py`
Implements the SVM model, including:
- Data preprocessing.
- One-hot encoding and feature scaling.
- Custom quadratic programming solver.
- Hard-margin SVM training and prediction.
- Evaluation of model performance.

---

## Preprocessing Steps

1. **Handle Missing Values**:
   - Impute missing values for numerical features using the median.
   - Skip rows with missing target variables (`Survived`).

2. **One-Hot Encoding**:
   - Convert categorical features (`Sex`, `Embarked`) into binary vectors.
   - Example:
     - `Sex = 'male'` → `[1, 0]`
     - `Sex = 'female'` → `[0, 1]`

3. **Feature Scaling**:
   - Standardize numerical features to have zero mean and unit variance.
   - One-hot encoded features remain untouched.

---

## Evaluation Metrics

The model's performance is evaluated using:
- **Accuracy**: Percentage of correct predictions.
- **Precision, Recall, F1-Score** (optional): For imbalanced datasets, these metrics provide a more nuanced view of performance.

Example output:
```
Hard-Margin SVM Accuracy: 78.50%
Precision: 0.80, Recall: 0.75, F1-Score: 0.77
```

---

## Limitations

1. **Hard-Margin Assumption**:
   - Assumes the data is linearly separable, which is rarely true for real-world datasets.
   - Consider implementing a soft-margin SVM for better generalization.

2. **No Kernel Support**:
   - The current implementation only supports linear kernels. Adding kernel functions (e.g., RBF, polynomial) would improve performance for non-linear data.

3. **Scalability**:
   - The custom QP solver may not scale well for large datasets. Using a more efficient optimization library (e.g., `cvxopt`) is recommended for larger problems.

4. **Missing Data Handling**:
   - The current implementation uses simple imputation (median). More advanced techniques (e.g., KNN imputation) could improve results.

---

## Future Improvements

1. **Soft-Margin SVM**:
   - Allow for some misclassifications by introducing slack variables.

2. **Kernelized SVM**:
   - Add support for non-linear kernels (e.g., RBF, polynomial).

3. **Cross-Validation**:
   - Implement k-fold cross-validation for more robust evaluation.

4. **Hyperparameter Tuning**:
   - Optimize hyperparameters like the regularization parameter $C$.

5. **Advanced Missing Data Handling**:
   - Use more sophisticated imputation methods (e.g., KNN, regression-based imputation).

--------