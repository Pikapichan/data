#映画の評価アンケートから自分が面白いと感じる映画を紹介する
!pip install gspread
from google.colab import auth
from google.auth import default
import gspread
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


# URLとシート名を指定
url = 'https://docs.google.com/spreadsheets/d/1ZPr0NkCrqVsUTb1-0lLW_MRNJknsF6qyujZ9jBYsMTk/edit?usp=sharing'
sheet = "Sheet1"

workbook = gc.open_by_url(url)
worksheet = workbook.worksheet(sheet)

column =worksheet.get_all_values()[0]
df = pd.DataFrame(worksheet.get_all_values()[1:], columns=column)   # 1行目を列名として、2行目以降を取得

!pip install kogi
import kogi.jwu

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
df.replace('', np.nan, inplace=True)
df = df.dropna()
matrix = df[['ワクワク', 'キュン', 'ドキドキ','ホッコリ', 'フムフム', 'ギャー', 'クスクス', 'スッキリ', 'モヤモヤ', 'ぐすん', '音楽', '映像']]

similarity_matrix = cosine_similarity(matrix.values)
print(similarity_matrix)

similarity_matrix[37]

df['評価'] = similarity_matrix[37]
df.head()

results = [(v, i) for i, v in enumerate(similarity_matrix[1])]
results.sort(reverse=True)

for v, i in results[0:10]:
    print(v, df.iloc[i]['映画タイトル'])

def sim(n):
    results = [(v, i) for i, v in enumerate(similarity_matrix[n])]
    results.sort(reverse=True)
    for v, i in results[0:10]:
        print(v, df.iloc[i]['映画タイトル'], "by", df.iloc[i]['推薦者(ペンネーム)'])
    print()

import matplotlib.pyplot as plt
import seaborn as sns
try:
    import japanize_matplotlib #matplotlibの日本語化
except ModuleNotFoundError:
    import os
    os.system('pip3 install japanize_matplotlib')
    import japanize_matplotlib
sns.set(font="IPAexGothic") #日本語フォント設定

from sklearn.linear_model import LinearRegression
model = LinearRegression()
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#自身が面白いと評価した映画を基準にコサイン類似度を求める
# 45行目以外の行を取得
df_selected_rows = df.drop(45)
# 5から16列目を取得
X = df_selected_rows.iloc[:, 5:17]

# 目的変数を設定する
y = df.iloc[45, 5:17]

print(X, y)
print(f'データ数: {len(df)}, 説明変数の次元 {X.shape[1]}')

from sklearn.feature_extraction.text import CountVectorizer

X=matrix
y = df['評価']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

# 残差プロットを描画
plt.figure(figsize=(3,3))
plt.scatter(y, y_pred, color='red', alpha=0.3)
plt.xlabel('実測')
plt.ylabel('予測')
plt.xlim(0, 1.5)
plt.ylim(0, 1.5)

mse = mean_squared_error(y, y_pred)
print('MSE:', mse)
r2 = r2_score(y, y_pred)
print('決定係数(R2):', r2)

y_pred = model.predict(X)
pd.DataFrame({'実測': df['評価'], '予測':y_pred}).head()

#映画の評価を打ち込むことで自身に合うかどうか判別
a=model.predict([(1,1,1,1,1,1,1,1,1,1,0,0)])

if 0.8<=a<1:
  print(a,'きっとこの映画が気に入るでしょう！')
elif 0.6<=a<0.8:
  print(a,'もしかしたら好きかもしれません')
else:
  print(a,'見ない方がいいかも・・・')
