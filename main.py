import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score , confusion_matrix, classification_report

print('Different Regression Comparison - 1k row dataset - test size=0.5')
# Calling the data
df = pd.read_csv('Dataset(100).csv')

# Separating the data into dependent (X) and independent (Y)
X = df[['load_P', 'cubed_L', 'inv_E', 'inv_I']]
Y = df['V_dis']

# splitting data 80% for traning and 20% for testing the model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=100)


# Traning the model1
reg = LinearRegression()
reg.fit(X_train, Y_train)
# Appling model to preedict y
Y_pred1 = reg.predict(X_test)
# Evaluating the model1
mse_1 = mean_squared_error(Y_test, Y_pred1)
R2_1 = r2_score(Y_test, Y_pred1)
print('MSE_Model1: ', mse_1)
print('R2_Model1: ', R2_1)

# Training the model2
regressor = DecisionTreeRegressor()
regressor.fit(X_train, Y_train)
Y_pred2 = regressor.predict(X_test)
# Evaluating the model2
mse_2 = mean_squared_error(Y_test, Y_pred2)
R2_2 = r2_score(Y_test, Y_pred2)
print('MSE_Model2: ', mse_2)
print('R2_Model2: ', R2_2)

# Training the model3
clf = RandomForestRegressor()
clf.fit(X_train, Y_train)
Y_pred3 = clf.predict(X_test)
# Evaluating the model2
mse_3 = mean_squared_error(Y_test, Y_pred3)
R2_3 = r2_score(Y_test, Y_pred3)
print('MSE_Model3: ', mse_3)
print('R2_Model3: ', R2_3)

# Data visualization

#Model1
polt_1 = plt.scatter(Y_test, Y_pred1, alpha= 0.3, color='#F6D55C', label='LinearRegression')
z_1 = np.polyfit(Y_test,Y_pred1,1)
Y_poly_1 = np.poly1d(z_1)
plt.plot(Y_test, Y_poly_1(Y_test), color='#F6D55C' , label= 'Best fit')

#Model2
polt_2 = plt.scatter(Y_test, Y_pred2, alpha= 0.3, color='#3CAEA3' , label='DecisionTreeRegressor')
z_2 = np.polyfit(Y_test,Y_pred2,1)
Y_poly_2 = np.poly1d(z_2)
plt.plot(Y_test, Y_poly_2(Y_test), color='#3CAEA3', label= 'Best fit')

#Model3
polt_3 = plt.scatter(Y_test, Y_pred3, alpha= 0.3, color='#ED553B' , label='RandomForestRegressor')
z_3 = np.polyfit(Y_test,Y_pred3,1)
Y_poly_3 = np.poly1d(z_3)
plt.plot(Y_test, Y_poly_3(Y_test), color='#ED553B', label= 'Best fit')

plt.legend()
plt.xlabel('The true value of V_dis (m)')
# Adjust the font size of the y-axis tick labels
plt.yticks(fontsize=5)  # Adjust the font size as needed
plt.ylabel('The predicted value of V_dis (m)')
plt.title('Different Regression Comparison - 1k row dataset - test size=0.5')
plt.show()


