import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import scipy.stats as stats

# takes data from obscure weird photo editing software (NOT FOR SCIENTFIC PURPOSES) from fucking 1995 
# that we had to use to "analyse" our microphotography images, it was a complete farce.
# didn't even used that data in my presentation because it was utterly bogus

def get_files(dir_path='D:\Downloads\csvs'):
    dir_path = dir_path
    res = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    return res


def yes_or_no(message):
    while True:
        print(message, '\n(1, y) - yes\n(0, n) - no')
        choice = input()
        if choice in ['0', '1', 'n', 'y']:
            break
        else:
            print('invalid input\n')
            pass
    return choice


res = get_files()
if not res:
    print('--no files found--')
results_df = pd.DataFrame()
for file in res:
    sep = '\t'
    conv = lambda x: (x.replace(",", "."))
    data = pd.read_csv(f'D:\Downloads\csvs\{file}', sep=sep)
    data = pd.read_csv(f'D:\Downloads\csvs\{file}', sep=sep, converters={data.columns[0]: conv, data.columns[-2]: conv})
    data = data[[data.columns[0], data.columns[-2]]]
    data = data.apply(pd.to_numeric)
    # 0 3 4 7 11 9
    data.columns = ['Частицы', 'Радиус, нм']
    data[data.columns[1]] *= 0.5
    radius_data = data[data.columns[1]]
    sample_mean = data[data.columns[1]].mean()
    sample_std = radius_data.std()
    alpha = 0.05
    degrees_of_freedom = len(radius_data) - 1
    critical_value = stats.t.ppf(1 - alpha / 2, df=degrees_of_freedom)
    margin_of_error = critical_value * (sample_std / (len(radius_data) ** 0.5))
    results = pd.DataFrame({'File': [file],
                            'Mean': [sample_mean],
                            'Median': [data[data.columns[1]].median()],
                            'Margin of Error': [margin_of_error],
                            'Standard Deviation': [data[data.columns[1]].std()],
                            'Confidence interval': [stats.t.interval(alpha=0.95, df=len(data[data.columns[1]])-1,
                                                                     loc=np.mean(data[data.columns[1]]),
                                                                     scale=stats.sem(data[data.columns[1]]))-sample_mean],
                            'Presentile 75': data[data.columns[1]].quantile(0.75),
                            'Presentile 25': data[data.columns[1]].quantile(0.25)})
    results_df = pd.concat([results_df, results]).reset_index(drop=True)
pd.set_option('display.max_columns', 20)
print(results_df)
