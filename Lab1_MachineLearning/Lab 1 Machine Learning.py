#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd


# In[14]:


xtrain = np.load('/Users/marianahenriques/Documents/X_train.npy')
ytrain = np.load('/Users/marianahenriques/Documents/y_train.npy')
xtest = np.load('/Users/marianahenriques/Documents/X_test.npy')


# In[15]:


data = pd.DataFrame({'X_1': xtrain[:,0], 'X_2': xtrain[:,1],'X_3': xtrain[:,2], 
                     'X_4': xtrain[:,3],'X_5': xtrain[:,4],'y': ytrain})


# In[111]:


arrays_equal = np.array_equal(xtrain, xtest)

print(arrays_equal)


# # Data Shape 

# In[10]:


np.shape(xtrain)


# In[19]:


np.shape(ytrain) #it's a vector 


# In[22]:


np.shape(xtest)


# In[145]:


data.hist() ##gaussian data, except the y 


# In[187]:


print(data.describe()) # count: Number of non-null values in each column; std: The standard deviation; 25% percentil: it's the point in the dataset where 25% of the values are less than or equal to it, and 75% of the values are greater than it


# In[184]:


sns.boxplot(ytrain)
plt.show()


# In[15]:


#Counting the number of missing values per line and compute the frequencies observed
aux = np.isnan(xtrain).sum(axis=1)
for i in np.unique(aux):
    print(i, np.sum(aux == i)) 
    


# # Outliers

# In[15]:


##doing our linear regression model with LinearRegression and testing with all our xtrain to see the outliers

from sklearn.linear_model import LinearRegression

# Initialize the linear regression model
model = LinearRegression()

# Fit the model using the training data
model.fit(xtrain, ytrain)

# Now you can make predictions using this model
ypred = model.predict(xtrain)

# Print model coefficients (beta values) and intercept
print("Coefficients (beta values):", model.coef_)
print("Intercept:", model.intercept_)

# Print first few predictions
#print(ypred)


# In[23]:


mean_squared_error(ytrain,ypred)


# In[50]:


def point_se(y1,y2):
    return (y1-y2)**2

list_errors = [(point_se(ytrainp,ypredp),i) for ytrainp, ypredp,i in zip(ytrain,ypred,list(range(len(xtrain))))]
strange_list_indices = [i for (errors,i) in list_errors if errors > 14]
len(strange_list_indices)


# In[20]:


xtrain_new = np.delete(xtrain,strange_list_indices, axis = 0)


# In[21]:


ytrain_new = np.delete(ytrain, strange_list_indices)


# In[22]:


(xtrain_new.shape,ytrain_new.shape)


# In[149]:


data_new = pd.DataFrame({'X_1': xtrain_new[:,0], 'X_2': xtrain_new[:,1],'X_3': xtrain_new[:,2], 
                     'X_4': xtrain_new[:,3],'X_5': xtrain_new[:,4],'y': ytrain_new})

data_new.hist()


# In[44]:


from sklearn.model_selection import train_test_split #val stands for validation

np.random.seed(2)
xtrain_new1, x_newval, ytrain_new1, y_newval = train_test_split(xtrain_new, ytrain_new, test_size = 0.2)


# In[45]:


from sklearn.linear_model import LinearRegression

# Initialize the linear regression model
model_new = LinearRegression()

# Fit the model using the training data
model_new.fit(xtrain_new1, ytrain_new1)

# Now you can make predictions using this model
ypred_new = model_new.predict(x_newval)

# Print model coefficients (beta values) and intercept
#print("Coefficients (beta values):", model_new.coef_)
#print("Intercept:", model_new.intercept_)

# Print first few predictions
#print(ypred_new)


# In[46]:


mean_squared_error(ypred_new,y_newval)


# In[47]:


# find the best alpha for Lasso regression
from sklearn.linear_model import LassoCV

# LassoCV automatically tunes alpha using cross-validation
model_cv = LassoCV(cv=5)  # 5-fold cross-validation

# Fit the model on the training data
model_cv.fit(xtrain_new1, ytrain_new1)

# Best alpha found by cross-validation
print("Best alpha:", model_cv.alpha_)

# Predictions on training and test data
y_pred_train_cv = model_cv.predict(x_newval)


# In[48]:


from sklearn.linear_model import Lasso

# Initializing the Lasso model with the regularization parameter alpha
model = Lasso(alpha=0.5027727306542653)
#0.5027727306542653


# Fitting the model on the training data
model.fit(xtrain_new1, ytrain_new1)

# To view the coefficients learned by the model, you can use:
#print("Lasso Coefficients:", model.coef_)

# To predict on training or test data, you can use:
y_pred_trainLASSO = model.predict(x_newval)


# In[49]:


mean_squared_error(y_pred_trainLASSO,y_newval)


# # Relationship between the predictors and the dependent variable

# In[190]:


correlation = data.corr()
plot = sns.heatmap(correlation, annot = True, fmt=".1f", linewidths=.6)
plot


# ## Minimum Covariance Determinant
# (CHI-SQUARED)

# In[108]:


import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def detect_outliers(X, threshold):
    mcd = MinCovDet().fit(X)
    mahal_distances = mcd.mahalanobis(X)
    outliers = mahal_distances > threshold
    return outliers, sum(outliers)

# Find the threshold for exactly 50 outliers
X =  data # Include y in the outlier detection
target_outliers = 50
mcd = MinCovDet().fit(X)
threshold_low = 0.0
threshold_high = np.max(mcd.mahalanobis(X))
max_iterations = 100
iteration = 0

while iteration < max_iterations:
    threshold = (threshold_low + threshold_high) / 2
    outliers, removed = detect_outliers(X, threshold)
    
    if removed == target_outliers:
        break
    elif removed < target_outliers:
        threshold_high = threshold
    else:
        threshold_low = threshold
    
    iteration += 1

print(f"Threshold: {threshold}")
print(f"Number of elements removed: {removed}")
print(f"Number of iterations: {iteration}")

# Clean the data
X_clean = X[~outliers]

# Split the data
#X_train, X_test, y_train, y_test = train_test_split(X_clean.drop(columns='y'), X_clean['y'], test_size=0.2, random_state=42)


X_train=X_clean.drop(columns='y')
y_train=X_clean['y']
# Train and evaluate the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)

mse = mean_squared_error(y_train, y_pred)
print(f"Mean Squared Error: {mse}")


# In[109]:


import optuna
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
# Define the objective function
def objective(trial, model_type):
    if model_type == 'Ridge':
        alpha = trial.suggest_loguniform('alpha', 1e-3, 1e3)
        model = Ridge(alpha=alpha)
    elif model_type == 'Lasso':
        alpha = trial.suggest_loguniform('alpha', 1e-3, 1e3)
        model = Lasso(alpha=alpha)
    elif model_type == 'ElasticNet':
        alpha = trial.suggest_loguniform('alpha', 1e-3, 1e3)
        l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elif model_type == 'HuberRegressor':
        epsilon = trial.suggest_uniform('epsilon', 1.0, 2.0)
        alpha = trial.suggest_loguniform('alpha', 1e-3, 1e3)
        model = HuberRegressor(epsilon=epsilon, alpha=alpha)
    elif model_type == 'LinearSVR':
        C = trial.suggest_loguniform('C', 1e-3, 1e3)
        epsilon = trial.suggest_uniform('epsilon', 0.0, 1.0)
        model = LinearSVR(C=C, epsilon=epsilon, random_state=42)
    else:
        raise ValueError("Unsupported model type")

    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_train)
    
    # Calculate the mean squared error
    mse = mean_squared_error(y_train, y_pred)
    
    return mse



# Dictionary to store results
results = {}

# Create and optimize studies for each model
for model_type in ['Ridge', 'Lasso', 'ElasticNet', 'HuberRegressor', 'LinearSVR']:
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model_type), n_trials=1000)
    
    # Store the results
    results[model_type] = {
        'mse': study.best_value,
        'best_params': study.best_params
    }

# Determine the best model
best_model_name = min(results, key=lambda k: results[k]['mse'])

# Print the best model details
print(f"Best model: {best_model_name}")
print(f"Best MSE: {results[best_model_name]['mse']}")
print(f"Best parameters: {results[best_model_name]['best_params']}")


# In[112]:


ridge_model = Ridge(alpha=0.0010007815019382835)

# Fit the model to the training data
ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
print(f"Mean Squared Error: {mse}")


# In[113]:


y_final = ridge_model.predict(xtest)


# In[115]:


np.save("/Users/marianahenriques/Documents/y_final.npy",y_final)


# In[116]:


y_final.shape


# In[ ]:




