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
Libraries are used for comparison only!
```bash
python data_loader.py
