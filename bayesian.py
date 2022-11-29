# Import libraries
import pickle

import matplotlib.pyplot as plt
import pymc as pm
from sklearn.datasets import fetch_california_housing

from model_utils import preprocess_housing

# Load California housing dataset

california_housing = fetch_california_housing(as_frame=True)
x = california_housing.data
print(x.shape)
y = california_housing.target
print(y.shape)

x

# Preprocess datapoints

X_train, X_test, y_train, y_test = preprocess_housing(x,y)

x = X_train
y = y_train

num_samples = 1000
model = pm.Model()

if __name__ == "__main__":
    with model:
        # Defining our priors
        w0 = pm.Normal('w0', mu=0, sigma=5)
        w1 = pm.Normal('w1', mu=0, sigma=5)
        w2 = pm.Normal('w2', mu=0, sigma=5)
        w3 = pm.Normal('w3', mu=0, sigma=5)
        w4 = pm.Normal('w4', mu=0, sigma=5)
        w5 = pm.Normal('w5', mu=0, sigma=5)
        w6 = pm.Normal('w6', mu=0, sigma=5)
        w7 = pm.Normal('w7', mu=0, sigma=5)
        w8 = pm.Normal('w8', mu=0, sigma=5)
        sigma = pm.Uniform('sigma', lower=0, upper=20)

        y_est = w0 + w1*x.loc[:,"MedInc"] + w2*x.loc[:,"HouseAge"] + w3*x.loc[:,"AveRooms"] + w4*x.loc[:,"AveBedrms"] + w5*x.loc[:,"Population"] + w6*x.loc[:,"AveOccup"] + w7*x.loc[:,"Latitude"] + w8*x.loc[:,"Longitude"]

        likelihood = pm.Normal('y', mu=y_est, sigma=sigma, observed=y)

        sampler = pm.NUTS()
        # sampler = pm.Metropolis()

        idata = pm.sample(num_samples, sampler, progressbar=True, cores=4, chains=4)

        pickle.dump(idata, open("sampler_inference.pkl", "wb"))
