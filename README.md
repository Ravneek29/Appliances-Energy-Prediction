# Appliances Energy Prediction

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Relevant Literature](#relevant-literature)
- [Contributors](#contributors)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [License](#license)

## Project Overview
This project aims to compare different types of Machine Learning algorithms to determine the best-performing model for predicting household energy consumption. The algorithms tested include Ridge Regression, Lasso Regression, Support Vector Regression, K-Neighbors Regression, Random Forest Regression, and Multi-Layer Perceptron Regression. The primary focus is to find the best hyperparameters for each algorithm to optimize prediction accuracy.

## Dataset
We used the **Appliances Energy Prediction Data Set** for this project. This dataset contains features related to temperature, humidity, and various other factors that influence the energy consumption of a household. The dataset meets the requirement of having at least 5,000 instances and 20 attributes, making it ideal for applying regression algorithms.

## Algorithms
The following algorithms were selected based on the dataset’s characteristics and the problem's complexity:

- **Ridge Regression:** Prevents overfitting in regression models.
- **Lasso Regression:** Reduces model complexity by shrinking coefficients.
- **Support Vector Regression (SVR):** Handles high-dimensional data effectively.
- **K-Neighbors Regression (KNN):** Captures local behavior of data and handles non-linear relationships.
- **Random Forest Regression:** An ensemble method that improves accuracy by combining multiple decision trees.
- **Multi-Layer Perceptron Regression:** A neural network-based approach that handles complex patterns in data.

## Installation
To set up this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/energy-prediction.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd energy-prediction
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To use this project, open the Jupyter Notebook `Appliances_Energy_Prediction.ipynb` in your preferred environment. The notebook contains the complete analysis, including data preprocessing, model training, and evaluation.

## Results
The project results include:

- **Support Vector Regression:** Achieved a training score of 0.173 and a testing score of 0.149 with the best hyperparameters.
- **Ridge Regression:** Achieved a training score of 0.138 and a testing score of 0.122.
- **K-Neighbors Regression:** Showed the best testing score of 0.576, making it the most effective model for this dataset.
- **Lasso Regression:** Improved testing score to 0.113 with hyperparameter tuning.
- **Random Forest Regression:** Testing score of 0.570, slightly lower than K-Neighbors Regression.
- **Multi-Layer Perceptron Regression:** Testing score of 0.382 with optimized hyperparameters.

These results show that K-Neighbors Regression performed best on this dataset.

## Relevant Literature
- **Regression Models:** Key in predicting scenarios using parameters like temperature and humidity. [9]
- **Support Vector Regression:** Efficient for high-dimensional data with margin errors. [5]
- **Ridge Regression:** Handles multiple correlated independent variables. [8]
- **K-Neighbors Regression:** Simple yet effective for non-parametric statistical predictions. [7]
- **Lasso Regression:** Uses L1 regularization for more accurate predictions. [3]
- **Random Forest Regression:** Ensemble learning method combining decision trees. [4]
- **Multi-Layer Perceptron Regression:** Neural network approach for complex data patterns. [6]


## Conclusion and Future Work
In conclusion, the K-Neighbors Regression algorithm performed best in predicting energy consumption based on the Appliances Energy Prediction dataset. Future work could involve exploring other models, such as Support Vector Machines, Artificial Neural Networks, or Gradient Descent algorithms, to improve accuracy further. Additionally, using a more advanced dataset with additional features like appliance usage could lead to better predictions.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
