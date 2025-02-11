# [Diabetes Detection Project](https://www.kaggle.com/code/hassaneskikri/diabetes-detection/notebook)

## Overview
This project focuses on the early detection of diabetes using advanced analytical methods. The aim is to develop a predictive model to identify high-risk individuals and facilitate timely medical interventions.

## Objectives
- Develop a robust predictive model for diabetes detection.
- Evaluate multiple machine learning algorithms.
- Optimize hyperparameters for improved performance using Optuna.
- Enhance healthcare outcomes through early detection.

## Methodology
### Model Comparison
Several classification algorithms were evaluated based on train and test accuracies:

| Model                  | Train Accuracy | Test Accuracy |
|------------------------|----------------|---------------|
| Logistic Regression    | 0.786585       | 0.756098      |
| Decision Tree          | 1.000000       | 0.731707      |
| Random Forest          | 1.000000       | 0.772358      |
| Gradient Boosting      | 0.865854       | 0.780488      |
| AdaBoost               | 0.849593       | 0.674797      |
| Support Vector Machine | 0.827236       | 0.772358      |
| K-Nearest Neighbors    | 0.815041       | 0.772358      |

### Final Model Selection & Hyperparameter Optimization
Based on the model comparison, the **Gradient Boosting Classifier** was chosen as the best model. Hyperparameters were fine-tuned using Optuna to achieve optimal performance. 

after hyperparameter optimization, the model achieved an accuracy of 0.8048780487804879 on the test set.and 0.9491869918699187 on the training set.

## Installation
1. Clone the repository.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
- Open the Jupyter Notebook located in the `Notebook/` directory:
   ```
   jupyter notebook Notebook/diabetes-detection.ipynb
   ```
- Follow the instructions in the notebook to explore the data, train models, and evaluate performance.

## Data Source
The dataset for this project is located at `Data/diabetes.csv` and contains clinical measurements and patient information.


## Conclusion
This project demonstrates a comprehensive approach to diabetes detection by:
- Evaluating multiple machine learning algorithms.
- Selecting Gradient Boosting as the final model.
- Optimizing hyperparameters using Optuna to enhance predictive performance.