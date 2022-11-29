from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

def learning_curves(mod, X_train, y_train , cv=5):
    N , train_score, val_score = learning_curve(mod, X_train, y_train,  cv=5 , train_sizes=np.linspace(0.2 ,1.0, 5))
    plt.plot(N, train_score.mean(axis=1), label='Train')
    plt.plot(N, val_score.mean(axis=1), label='Validation')
    plt.xlabel('train size')
    plt.legend()
    
def preprocess_housing(x, y):
    if x.isnull().sum()[1] > 0:
        print("null values need to be removed")
    else:
        print("data has no null values")
        
    rmv_outlier_data_x = x[(np.abs(zscore(x)) < 3).all(axis=1)]
    rmv_outlier_data_y = y[(np.abs(zscore(x)) < 3).all(axis=1)]
    print("Removed", x.shape[0] - rmv_outlier_data_x.shape[0], "outlier rows")
    
    print("Target value counts:", y.value_counts().head(1))
    rmv_high_vals_x = rmv_outlier_data_x[rmv_outlier_data_y != 5.00001]
    rmv_high_vals_y = rmv_outlier_data_y[rmv_outlier_data_y != 5.00001]
    print("Removed", rmv_outlier_data_x.shape[0] - rmv_high_vals_x.shape[0], "skewed rows")

    X_train, X_test, y_train, y_test = train_test_split(rmv_high_vals_x, rmv_high_vals_y)
    return X_train, X_test, y_train, y_test