!wget https://kkuramitsu.github.io/lec/data/bento.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import japanize_matplotlib #matplotlibの日本語化  
except ModuleNotFoundError:
    !pip install japanize_matplotlib
    import japanize_matplotlib 
sns.set(font="IPAexGothic") #日本語フォント設定
!pip install -q kogi
import kogi

df = pd.read_csv('bento.csv')
df.head()
df.dropna(axis=1)
df['kcal'].describe()
df['kcal'] = df['kcal'].fillna(df['kcal'].mean()).astype(int)
df.head()
df['payday'] = df['payday'].fillna(0)
df['event'] = df['event'].fillna('なし')
df['remarks'] = df['remarks'].fillna('特になし')
df.head()
print(df['name'])df['weather'].map({
    '快晴': 0, '晴れ': 1, '薄曇': 2, '曇': 3, '雨': 4, '雷電': 5, '雪': 6
})

df_weather = pd.get_dummies(df['weather'], dummy_na=False, columns=['weather'])
df_weather.head()

df_weather = pd.get_dummies(df['weather'], dummy_na=False, columns=['weather'])
df_weather.head()

df.groupby('weather')['y'].median()

#天気と売上の関係
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df[['temperature']]
y = df['y']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print('説明変数:', X.columns, '次元:', X.shape)
print("MSE: ", mean_squared_error(y, y_pred))
print('R2', r2_score(y, y_pred))

plt.figure(figsize=(7, 7))
plt.scatter(y, y_pred, c='red', alpha=0.3)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

print('訓練データ数:', len(X_train))
print('テストデータ数:', len(X_test))

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE(テスト): ", mean_squared_error(y_test, y_pred) ) 
print('R2(テスト):', r2_score(y_test, y_pred))
print('R2(訓練):', r2_score(y_train, model.predict(X_train)))

#すべて変数化して売り上げを予測
from sklearn.preprocessing import LabelEncoder
for i in df.columns:
  week_le = LabelEncoder()
  df[i] = week_le.fit_transform(df[i])

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df[df.columns[2:]]
y = df['y']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print('説明変数:', X.columns, '次元:', X.shape)
print("MSE: ", mean_squared_error(y, y_pred))
print('R2', r2_score(y, y_pred))

plt.figure(figsize=(7, 7))
plt.scatter(y, y_pred, c='red', alpha=0.3)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

print('訓練データ数:', len(X_train))
print('テストデータ数:', len(X_test))

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE(テスト): ", mean_squared_error(y_test, y_pred) ) 
print('R2(テスト):', r2_score(y_test, y_pred))
print('R2(訓練):', r2_score(y_train, model.predict(X_train)))

#決定木
from sklearn.tree import plot_tree
plot_tree(model, feature_names=X.columns, filled=True)
plt.show()
import xgboost as xgb
model = xgb.XGBRegressor(objective ='reg:squarederror')
model.fit(X_train,y_train) 
xgb.plot_importance(model)
xgb.to_graphviz(model, num_trees=5)
