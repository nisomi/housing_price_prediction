## Housing Price Prediction

This project is developed to predict housing prices based on various characteristics. It uses the XGBoost Regressor ensemble model, which allows for accurate predictions.

### Requirements
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, xgboost

### Installation
1. Clone the repository: `git clone https://github.com/nisomi/housing_price_prediction.git`
2. Install the required libraries: `pip install -r requirements.txt`

### Usage
1. Prepare training and testing data in CSV format.
2. Run `model_train.py` to train the model.
3. After training the model, execute `model_test.py` to make predictions.

### Project Structure
- `data/`: Directory for data
- `model/`: Directory for storing the trained model
- `pipeline/`: Directory for model code and data preprocessing
- `src/`: Directory for source code
- `requirements.txt`: File with required libraries

### Results
The XGBoost Regressor model was trained and tested with the following metrics:

- Mean Absolute Error: 76430.82
- Mean Squared Error: 12986694348.95
- Root Mean Squared Error: 113959.18
- R^2 Score: 0.74

