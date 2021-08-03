#... Let's import our libraries..

import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib

house = pd.read_csv(r"C:\Users\xingx\OneDrive\Desktop\Data Science\Machine Learning\Maison.csv")

# Since the columns are in french, in order to make them more readable,
#  let's translate them into English

house = house.rename(index = str, columns = {'PRIX':'price','SUPERFICIE': 'area','CHAMBRES': 'rooms',
                            'SDB': 'bathroom', 'ETAGES': 'floors','ALLEE': 'driveway',
                             'SALLEJEU':'game_room', 'CAVE': 'cellar', 
                             'GAZ': 'gas', 'AIR':'air', 'GARAGES': 'garage', 'SITUATION': 'situation'})

#...Looking at the data after translating it from French to English..
house.head()

#....Checking the shape of the data set...
house.shape

#.....Checking for datatypes...
house.info()

#...Checking for null values....
house.isnull().sum()

# Checking for correlation among columns to see which columns are highly correlated
house.corr()

#.....Checking for statistical data of each variable in the data set...
house.describe()

#....Data set after Outlier Removal....
house

# Standardizing the data to normal distribution
    
data = house.iloc[0:11].values
from sklearn import preprocessing
house_standardized = preprocessing.scale(data)
house_standardized = pd.DataFrame(house_standardized)
print(house_standardized)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#.. We now instantiate a Linear Regression object...
    
lm = LinearRegression()

#....Let's do the split of the dataset....
    
house.columns
X = house[['area', 'rooms','bathroom','floors', 'driveway', 'game_room',
           'cellar', 'gas', 'air', 'garage', 'situation']]
y = house['price']

#....Labelling the training and testing data.....
    
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size= 0.2, random_state= 1)

#.. Let's check the head of some of these splits
X_test.head()

#...Now let's build the model using sklearn
lm.fit(X_test,y_test)
print('[info] model has been trained')

# accuracy
result = lm.score(X_test , y_test)
print(f'accuracy of the model is {result}')

#.... Now let's look at the coefficients...
coef = pd.DataFrame(lm.coef_, X.columns, columns = ['Coefficients'])

#....Putting the coefficients in a dataframe...

coef = pd.DataFrame(lm.coef_, X.columns, columns = ['Coefficients'])
coef


import statsmodels.api as sm

# Unlike sklearn that adds an intercept to our data for the best fit, statsmodel doesn't. We need to add it ourselves\n",
# Remember, we want to predict the price based off our features.\n",
# X represents our predictor variables, and y our predicted variable.\n",
# We need now to add manually the intercepts\n",
    
X_endog = sm.add_constant(X_test)

res = sm.OLS(y_test, X_endog)
res.fit()

res.fit().summary()

predictions = lm.predict(X_test)

#....Evaluation metrics....

# Mean Absolute Error (MAE)
# Mean Squared Error (MSE)
# Root Mean Squared Error(RMSE)

from sklearn import metrics

print('MAE :', metrics.mean_absolute_error(y_test, predictions))
print('MSE :', metrics.mean_squared_error(y_test, predictions))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

X_endog_test = sm.add_constant(X_test)
model = res.fit()
predictions = model.predict(X_endog_test)

predictions

# saving the model
joblib.dump(lm , 'maison.pkl')

# load the model
loaded_model = joblib.load('maison.pkl')

pred = loaded_model.predict([[3000, 234, 3, 2, 4, 2, 3, 4, 2, 1, 1]])
print(pred)







