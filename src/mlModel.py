import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import seaborn as sns
import gradio as gr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import warnings
import re
warnings.filterwarnings('ignore')

seed = 42
np.random.seed(seed)

df_train=pd.read_csv("https://raw.githubusercontent.com/s669799/DAT158-MLproject/refs/heads/main/assets/train.csv", index_col="id")
df_test=pd.read_csv("https://raw.githubusercontent.com/s669799/DAT158-MLproject/refs/heads/main/assets/test.csv")

# Sum of missing
missing_train_sum = df_train.isna().sum()
missing_test_sum = df_test.isna().sum()
# Percentage of missing data
missing_train_percent = df_train.isna().mean() * 100
missing_test_percent = df_test.isna().mean() * 100

# Filter rows where 'fuel_type' is NaN, and that are also actually electric
def set_electric_fuel_type(df):
    fuel_nan_df = df[df['fuel_type'].isnull()]
    non_electric_indicators = [
        'v6', 'v8', 'litre', 'liter', 'twin turbo', 'turbo',
        r'\d+[,.]\d+[lL]']
    pattern = '|'.join(non_electric_indicators)
    is_probably_non_electric = fuel_nan_df['engine'].str.contains(pattern, case=False, na=False)
    probably_electric_df = fuel_nan_df[~is_probably_non_electric]
    df.loc[probably_electric_df.index, 'fuel_type'] = 'Electricity'

    return df

df_train = set_electric_fuel_type(df_train)
df_test = set_electric_fuel_type(df_test)

# Handling missing values: Assume title not clean as seller does not have proof. 'No' if it's not filled.
df_train['clean_title'].fillna('No', inplace=True)
df_test['clean_title'].fillna('No', inplace=True)

df_train['accident'].fillna('None reported', inplace=True)
df_test['accident'].fillna('None reported', inplace=True)

df_train.head()

def extract_engine_features(df):
  # Extract horsepower
  df['horsepower'] = df['engine'].apply(lambda x: float(re.search(r'(\d+(\.\d+)?)HP', x).group(1)) if re.search(r'(\d+(\.\d+)?)HP', x) else None)

  # Extract displacement
  df['displacement'] = df['engine'].apply(lambda x: float(re.search(r'(\d+\.\d+)L|(\d+\.\d+) Liter', x).group(1) or re.search(r'(\d+\.\d+)L|(\d+\.\d+) Liter', x).group(2)) if re.search(r'(\d+\.\d+)L|(\d+\.\d+) Liter', x) else None)

  # Extract engine type
  df['engine_type'] = df['engine'].apply(lambda x: re.search(r'(V\d+|I\d+|Flat \d+|Straight \d+)', x).group(1) if re.search(r'(V\d+|I\d+|Flat \d+|Straight \d+)', x) else None)

  # Extract Cylinder Count
  df['cylinders'] = df['engine'].apply(
      lambda x:
      int(re.search(r'\b(\d+) Cylinder\b', x).group(1))
      if re.search(r'\b(\d+) Cylinder\b', x)
      else (int(re.search(r'\b(V\d+|I\d+|Flat \d+|Straight \d+)\b', x).group(0)[1:])
            if re.search(r'\b(V\d+|I\d+|Flat \d+|Straight \d+)\b', x)
            else None)
    )

  # Extract Fuel Type
  fuel_types = ['Gasoline', 'Diesel', 'Electric', 'Hybrid', 'Flex Fuel']
  df['fuel_type'] = df['engine'].apply(lambda x: next((fuel for fuel in fuel_types if fuel in x), None))

  return df

df_train = extract_engine_features(df_train)
df_test = extract_engine_features(df_test)


missing_train = df_train.isna().mean() * 100
missing_test = df_test.isna().mean() * 100


# We have extraced what we need from engine, model is hard to categorize and engine_type has too much missing data to use effectively, so we drop these columns.
columns_to_drop = ['model', 'engine_type', 'engine']
df_train.drop(columns=[col for col in columns_to_drop if col in df_train.columns], inplace=True)
df_test.drop(columns=[col for col in columns_to_drop if col in df_test.columns], inplace=True)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

# Iterative Imputer: Limit max iterations and increase tolerance
imputer = IterativeImputer(max_iter=10, tol=1e-3, random_state=0)

df_train[['horsepower', 'displacement']] = imputer.fit_transform(df_train[['horsepower', 'displacement']])
df_test[['horsepower', 'displacement']] = imputer.transform(df_test[['horsepower', 'displacement']])

simple_imputer = SimpleImputer(strategy='mean')
df_train['cylinders'] = simple_imputer.fit_transform(df_train[['cylinders']])
df_test['cylinders'] = simple_imputer.transform(df_test[['cylinders']])

df_train.head()

# Transmission types
transmission_counts = df_train['transmission'].value_counts()
# print(transmission_counts)

def map_transmission(transmission):
    # Standardize the input
    transmission = transmission.strip().lower()

    if any(keyword in transmission for keyword in ['a/t', 'automatic']):
        return 'Automatic'
    elif any(keyword in transmission for keyword in ['m/t', 'manual']):
        return 'Manual'
    elif any(keyword in transmission for keyword in ['cvt', 'variator']):
        return 'Variator'
    elif any(keyword in transmission for keyword in ['dt', 'dual']):
        return 'Dual Clutch'
    else:
        return 'Other'

# Apply the function to the DataFrame
df_train['transmission'] = df_train['transmission'].apply(map_transmission)
df_test['transmission'] = df_test['transmission'].apply(map_transmission)

#print(df_train['transmission'].value_counts())
#print("\n")
#print(df_test['transmission'].value_counts())


def mapping_columns(df):
    # Replace values in the 'accident' column
    df["accident"] = df["accident"].replace({
        'At least 1 accident or damage reported': 1,
        'None reported': 0
    })

    # Replace values in the 'clean_title' column
    df["clean_title"] = df["clean_title"].replace({
        "Yes": 1,
        "No": 0
    })

    # Replace values in the 'transmission' column
    df["transmission"] = df["transmission"].replace({'Automatic':1, 'Dual Clutch':2, 'Manual':3, 'Variator':4, 'Other':5})

    return df

df_train = mapping_columns(df_train)
df_test = mapping_columns(df_test)

#plt.figure(figsize=(10, 4))
#sns.barplot(x='brand', y='price', data=df_train[:10000], errorbar=None)
#plt.title('Average Price by Car Brand')
#plt.xlabel('Brand')
#plt.ylabel('Average Price')
#plt.xticks(rotation=90)
#plt.show()

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Encoding categorical variables
categorical_columns = ['brand', 'fuel_type', 'ext_col', 'int_col']
lb = LabelEncoder()

for col in categorical_columns:
    if col in df_train.columns:
        df_train[col] = lb.fit_transform(df_train[col])
        df_test[col] = lb.transform(df_test[col])


df_train.head()

#plt.figure(figsize=(10, 8))
sns.heatmap(df_train.corr(), annot=True, cmap='inferno', fmt='.2f')
#plt.title('Correlation Heatmap')
#plt.show()

# Log (Natural logarithm) transform the highly skewed target variable (price)
df_train['price'] = np.log1p(df_train['price'])

#plt.figure(figsize=(10, 6))
sns.histplot(df_train['price'], kde=True, color='blue')
#plt.title('Price Distribution')
#plt.xlabel('Price')
#plt.show()

# Set the number of rows and columns for the grid
n_rows = 4
n_cols = 4

# Filter numerical columns only
num_columns = df_train.select_dtypes(include=['float64', 'int64']).columns

# Create the figure and axes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))

# Flatten axes array to easily iterate over
axes = axes.flatten()

# Plot each numeric column in a subplot
for i, col in enumerate(num_columns):
    sns.histplot(df_train[col], kde=True, ax=axes[i], color='blue')
    axes[i].set_title(f'Distribution of {col}', fontsize=12)

# If there are more subplots than features, remove the extra subplots
for j in range(len(num_columns), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
#plt.tight_layout()


X = df_train.drop(['price'],axis=1)
y = df_train['price']
X.shape, y.shape

# Initialize scaler
scaler = StandardScaler()

# Scale continuous features
continuous_features = ['milage', 'horsepower', 'displacement', 'cylinders', 'model_year']
df_train[continuous_features] = scaler.fit_transform(df_train[continuous_features])

# Scale the same way for the test set
df_test[continuous_features] = scaler.transform(df_test[continuous_features])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.225, random_state=42)

from sklearn.ensemble import RandomForestRegressor
# Define a reduced parameter grid
param_dist_rf = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False]
}

# Initialize the model
random_forest = RandomForestRegressor(random_state=42)

# Random search over the parameter grid with parallel processing
random_search_rf = RandomizedSearchCV(
    estimator=random_forest,
    param_distributions=param_dist_rf,
    n_iter=10,  # Reduced number of iterations
    scoring='neg_mean_squared_error',
    cv=3,  # Consider reducing cross-validation folds
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit the model
random_search_rf.fit(X_train, y_train)
best_params_rf = random_search_rf.best_params_
print("Best Parameters for Random Forest:", best_params_rf)

# Predict on the test set
y_pred = random_search_rf.predict(X_val)

# Evaluate the model
rmse = np.sqrt(np.mean((np.expm1(y_pred) - np.expm1(y_val))**2))

print(f'Optimized Random Forest RMSE: {rmse:.2f}')


import joblib
joblib.dump(random_search_rf, 'random_forest_model.pkl')
joblib.dump(lb, 'encoder.pkl')
