import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('gld_price_data.csv')
df.head()

df.dropna(inplace=True)

features = df.drop(['GLD', 'Date'], axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
scaled_df['GLD'] = df['GLD'].values

scaled_df.hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sns.heatmap(df.drop('Date', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title('Feature correlation matrix')
plt.show()

X = scaled_df.drop('GLD', axis=1)
y = scaled_df['GLD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

def evaluate_regression(y_true, y_pred, model_name):
    print(f"\n{model_name} evaluation:")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("R² :", r2_score(y_true, y_pred))

evaluate_regression(y_test, lr_preds, "Linear regression")
evaluate_regression(y_test, rf_preds, "Random forest regressor")

def categorize(price):
    if price < df['GLD'].quantile(0.33):
        return 'low'
    elif price < df['GLD'].quantile(0.66):
        return 'medium'
    else:
        return 'high'

df['price_level'] = df['GLD'].apply(categorize)

X_cls = scaled_df.drop('GLD', axis=1)
y_cls = df['price_level']
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_cls, y_train_cls)
log_preds = log_model.predict(X_test_cls)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train_cls, y_train_cls)
tree_preds = tree_model.predict(X_test_cls)

def evaluate_classification(y_true, y_pred, model_name):
    print(f"\n{model_name} classification report:")
    print(classification_report(y_true, y_pred))

evaluate_classification(y_test_cls, log_preds, "Logistic regression")
evaluate_classification(y_test_cls, tree_preds, "Decision tree classifier")

importances = rf_model.feature_importances_
plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=X.columns)
plt.title('Feature importance (random forest)')
plt.xlabel('Importance score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
