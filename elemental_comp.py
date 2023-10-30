import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#predicts and plots elemental composition based on avalible data

df_yield_1 = pd.DataFrame({'t': [320, 320, 730, 730, 1090, 1140, 1140, 1140],
                           'T': [1, 2, 1, 2, 1, 1, 2, 0],
                           'Выход, %': [46, 40, 35.33, 35.33, 34.67, 32, 32, 33]})

df_yield_2 = pd.DataFrame({'t': [320, 320, 730, 730, 1140, 1140, 1140],
                           'T': [1, 2, 1, 2, 1, 2, 0],
                           'Выход, %': [55.33,44.67,42.67,33.33,33.33,31.33,33.50]})

df_h_1 = pd.DataFrame({'t': [320, 320, 730, 730, 1090, 1140, 1140],
                     'T': [1, 2, 1, 2, 1, 1, 2],
                     'H, %': [0.46, 0.3, 0.02, 0.02, 0.01, 0, 0]})

df_h_2 = pd.DataFrame({'t': [320, 320, 730, 730, 1140, 1140],
                     'T': [1, 2.5, 1, 2.5, 1, 2.5],
                     'H, %': [2.71, 0.3, 0.07, 0.01, 0.01, 0]})

df_c_1 = pd.DataFrame({'t': [320, 320, 730, 730, 1090, 1140, 1140, 1140],
                     'T': [1, 2, 1, 2, 1, 1, 2, 0],
                     'C, %': [6.98, 3.23, 0.1, 0.27, 1.16, 0.06, 0.02, 0.09]})

df_c_2 = pd.DataFrame({'t': [320, 320, 730, 730, 1140, 1140, 1140],
                     'T': [1, 2.5, 1, 2.5, 1, 2.5, 0],
                     'C, %': [26.65, 7.16, 6.24, 0.81, 0.05, 0, 0.07]})
                     
df_mags = pd.DataFrame({'t': [320, 320, 730, 730, 1140],
                     'T': [1, 2, 1, 2, 1],
                     '\u03C3\u209B, %': [27.1, 47.4, 45.1, 41.9, 12.8]})

df_magr = pd.DataFrame({'t': [320, 320, 730, 730, 1140],
                     'T': [1, 2, 1, 2, 1],
                     '\u03C3\u1D63, %': [4.28, 10.9, 13.5, 12.3, 1.23]})

df_magh = pd.DataFrame({'t': [320, 320, 730, 730, 1140],
                     'T': [1, 2, 1, 2, 1],
                     'Hc, %': [121, 166, 256, 218, 242]})

df_append = None

# yield_1 m
# [0.69,0.6,0.53,0.53,0.52,0.48,0.49]

# yield_1 %
# [46,40,35.33,35.33,34.67,32,32.67]

# yield_2 m
# [0.80, 0.67, 0.64, 0.50, 0.5 , 0.47, 0.55]

# yield_2 %
# [55.33,44.67,42.67,33.33,33.33,31.33,33.50]

compound = 'Оксалат железа'

df_cust = pd.DataFrame({'t': [320, 320, 730, 730],
                        'T': [1, 2, 1, 2],
                        'Выход, %': [22, 24.67, 19.73, 15.3]})
######################################################################################
df = df_h_1
df = df.append(df_append)
degree = 2

#drop bad 1090
#df = df.drop(df[df['t'] == 1090].index)
search = df.columns[2]
features = df.drop([search], axis=1)
target = df[search]

poly = PolynomialFeatures(degree=degree, include_bias=False)
poly_features = poly.fit_transform(features)

model = LinearRegression()
model.fit(poly_features, target)


model_reg = ( f'у = {model.intercept_:.3} {model.coef_[0]:+.3}t {model.coef_[1]:+.3}\u03C4'
      f' {model.coef_[2]:+.3}t\u00B2 {model.coef_[3]:+.3}t\u03C4 {model.coef_[1]:+.3}\u03C4\u00B2')
title = f'{compound}, {search}\n{model_reg}'
print('Коэффициенты модели:', title)
x = np.linspace(200, 1200, 20)
y = np.linspace(1, 3, 20)
x, y = np.meshgrid(x, y)
predict = pd.DataFrame({'t': x.ravel(),
                        'T': y.ravel()})
poly_predict = poly.fit_transform(predict)
features_predict = np.column_stack([x.ravel(), y.ravel()])
B = model.predict(poly_predict)
if df.columns[2] != 'Выход, %':
    B[B < 0] = 0
#parts where t is above n = 0
    B[np.where(features_predict[:, 0] > 900)] /= 1.5
    B[np.where(features_predict[:, 0] > 950)] /= 3
    B[np.where(features_predict[:, 0] > 1000)] = 0
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')
#add scatter points
#ax.scatter(features['t'], features['T'], target)
ax.view_init(elev=20, azim=20)
ax.plot_trisurf(features_predict[:, 0], features_predict[:, 1], B, alpha=0.5)

ax.set_xlabel('t, \u00B0C')
ax.set_ylabel('\u03C4, час')
ax.set_zlabel(df.columns[2])
ax.set_title(title)
plt.savefig(f'D:\Downloads\csvs\plots\{compound}, {search}, plot.png')
data = pd.DataFrame({'t': features_predict[:, 0].astype('int'),
                         'T': features_predict[:, 1].round(2),
                         df.columns[2]: B.round(2)})
                     # FOR INT!
#                         'T': features_predict[:, 1].round(1),
#                         df.columns[2]: B.round(0)})
plt.figure(figsize=(11, 9))
(
# FOR NUMS
#sns.heatmap(data.pivot('t', 'T', df.columns[2]), annot=True, fmt='g')
sns.heatmap(data.pivot('t', 'T', df.columns[2]), fmt='g')
   .set(title=title,
        xlabel='\u03C4, час',
        ylabel='t, \u00B0C')
)
plt.savefig(f'D:\Downloads\csvs\plots\{compound}, {search}, heatm.png')