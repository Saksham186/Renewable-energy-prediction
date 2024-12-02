import pandas as pd
import kaggle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib
kaggle.api.authenticate()

dataset_name = "alistairking/renewable-energy-consumption-in-the-u-s"
destination = "/Users/anand/.cache/kagglehub/datasets/alistairking/renewable-energy-consumption-in-the-u-s/versions/1/"

kaggle.api.dataset_download_files(dataset_name, path=destination, unzip=True)
path = "/Users/anand/.cache/kagglehub/datasets/alistairking/renewable-energy-consumption-in-the-u-s/versions/1/dataset.csv"


data = pd.read_csv(path)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'


data['Season'] = data['Month'].apply(get_season)


season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
data['Season_Code'] = data['Season'].map(season_mapping)
data['Year_Trend'] = data['Year'] - data['Year'].min()
data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)


#print(data.head())
#print(data.info())
#print(data.describe())
#print(data.isnull().sum())
#print(data.duplicated().sum())
#sns.boxplot(x=data['Hydroelectric Power'])
#plt.show()

features = data[['Year_Trend', 'Season_Code', 'Month_Sin', 'Month_Cos', 'Hydroelectric Power', 'Solar Energy']]
target = data['Total Renewable Energy']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

'''print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")'''

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
grid_search.fit(X_train, y_train)

#print(f"Best Parameters: {grid_search.best_params_}")new_data = pd.DataFrame({

new_data = pd.DataFrame({
    'Year_Trend': [50],   # 50 years after the starting year
    'Season_Code': [2],   # Summer
    'Month_Sin': [0.866], # Sine of Month
    'Month_Cos': [-0.5],  # Cosine of Month
    'Hydroelectric Power': [1.0],
    'Solar Energy': [0.5]
})
prediction = model.predict(new_data)
#print(f"Predicted Total Renewable Energy: {prediction[0]}")
feature_importances = model.feature_importances_
features = X_train.columns

'''plt.barh(features, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance Analysis')
plt.show()'''
joblib.dump(model, 'energy_forecasting_model.pkl')