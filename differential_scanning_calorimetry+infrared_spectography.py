import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

print('imported libraries')


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


# res = res[:1]
def plotting(sep, file):
    conv = lambda x: (x.replace(",", "."))
    data = pd.read_csv(f'D:\Downloads\csvs\{file}', sep=sep)
    data = pd.read_csv(f'D:\Downloads\csvs\{file}', sep=sep, converters={data.columns[0]: conv, data.columns[1]: conv})
    data.columns = data.iloc[0]
    data = data.drop(0)
    data = data.apply(pd.to_numeric)
    # more than one A columns
    if len(data.columns) > 2:
        data['A'] = data[data.columns[1:]].mean(axis=1)
    # micrometers to cm-1
    if data[data.columns[0]].mean() < 100:
        data[data.columns[0]] = 1 / (data[data.columns[0]] * 10 ** -4)
    print('mean', data['A'].mean())
    choice = yes_or_no(f'{file[:-4]}, transform data to absorbance?')
    if choice in ['1', 'y']:
        data['A'] = data[data.columns[1:]].mean(axis=1)
        data['A'] = 2 - np.log10(data['A'] * 100)
        print('mean', data['A'].mean(), 'transformed')
    data.plot(x=data.columns[0],
              y=data.columns[-1],
              grid=True,
              title=file[:-4],
              legend=False,
              figsize=(10, 7),
              ylim=(0, None),
              xlim=(400, 4000),
              xlabel='\u03BD, см\u207B\u00B9',
              ylabel='Поглощение, отн. ед.')
    plt.gca().invert_xaxis()
    indices = find_peaks(data['A'], prominence=(0.001, 2))
    x_diff = 80
    y_diff = 0.007
    p_prev = 0
    idx_prev = 0
    cond = pd.DataFrame({'p': [],
                         'cond': []})
    for p in data[data.columns[0]][indices[0]]:
        idx = np.abs(data[data.columns[0]] - p).argmin()
        if p_prev > 0:
            if p > p_prev - x_diff and (data['A'][idx_prev] + y_diff > data['A'][idx] > data['A'][idx_prev] - y_diff):
                skip = pd.DataFrame({'p': [p],
                                     'cond': ['passed']})
                cond = pd.concat([cond, skip])
                continue
        plt.vlines(x=p, ymin=0, ymax=data['A'][idx], color='r', linestyle='--', linewidth=1)
        plt.text(p, data['A'][idx], str(int(p)), ha='center', va='bottom')
        p_prev = p
        idx_prev = idx
        add = pd.DataFrame({'p': [p],
                            'cond': ['added']})
        cond = pd.concat([cond, add])
    plt.savefig(f'D:\Downloads\csvs\plots\{file}.png')
    print(file[:-4], '\n', cond)
    # plt.show()


while True:
    while True:
        print('What\'s plotting?\n1 - IR \n2 - DSC')
        choice = input()
        if choice in ['1', '2']:
            break
        else:
            print('invalid input\n')
            pass
    if choice == '1':
        for file in res:
            try:
                plotting('\t', file)
                continue
            except (pd.errors.ParserError, ValueError) as e:
                print(f'--ParserError/ValueError, try formatting file-- {file[:-4]}')
            except:
                pass
            try:
                plotting(';', file)
                continue
            except (pd.errors.ParserError, ValueError) as e:
                print(f'--ParserError/ValueError, try formatting file-- {file[:-4]}')
            except:
                pass
            try:
                plotting(' ', file)
                continue
            except (pd.errors.ParserError, ValueError) as e:
                print(f'--ParserError/ValueError, try formatting file-- {file[:-4]}')
            except:
                pass
    if choice == '2':
        for file in res:
            conv = lambda x: (x.replace(",", "."))
            data = pd.read_csv(f'D:\Downloads\csvs\{file}', sep=';')
            data = pd.read_csv(f'D:\Downloads\csvs\{file}', sep=';',
                               converters={data.columns[1]: conv, data.columns[3]: conv})
            data = data.apply(pd.to_numeric)
            data['Тепловой поток (W/g)'] = data[data.columns[3]] / data[data.columns[2]]
            data['Вес (%)'] = data[data.columns[2]] / data[data.columns[2]].max() * 100
            ex = data.columns[1]
            uy_1 = data.columns[-2]
            uy_2 = data.columns[-1]
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('Температура, (\u00B0C)')
            ax1.set_ylabel(uy_1, color=color)
            ax1.plot(data[ex], data[uy_1], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            plt.xlim((0, 520))
            plt.ylim((data[uy_1].min(), 10))
            indices_1 = find_peaks(data[uy_1], prominence=(1, 50))
            for p in data[ex][indices_1[0]]:
                idx = np.abs(data[ex] - p).argmin()
                plt.vlines(x=p, ymin=data[uy_1].min(), ymax=data[uy_1][idx], color='r', linestyle='--', linewidth=1)
                plt.text(p, data[uy_1][idx], f'{p:.2f}', ha='center', va='bottom', size='small', color='tab:blue')
            indices_2 = find_peaks(-data[uy_1], prominence=(1, 50))
            for p in data[ex][indices_2[0]]:
                idx = np.abs(data[ex] - p).argmin()
                plt.vlines(x=p, ymin=data[uy_1].min(), ymax=data[uy_1][idx], color='r', linestyle='--', linewidth=1)
                plt.text(p, data[uy_1][idx] - 1.5, f'{p:.2f}', ha='center', va='bottom', size='small', color='tab:blue')
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel(uy_2, color=color)
            ax2.plot(data[ex], data[uy_2], color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            point = data[uy_2][indices_1[0][0]]
            delta = 100 - point
            plt.text(data[ex][indices_1[0][0]] - 20, point + delta / 2 - 2, f'{delta:.2f}',
                     ha='center', va='bottom', size='small', color='tab:red')
            point = data[uy_2][indices_1[0][1]]
            delta = 100 - point
            plt.text(data[ex][indices_1[0][1]] + 20, point + delta / 2 - 2, f'{delta:.2f}',
                     ha='center', va='bottom', size='small', color='tab:red')
            plt.ylim((40, 105))
            plt.axhline(y=100, linestyle='-', linewidth=1, color='tab:red')
            plt.savefig(f'D:\Downloads\csvs\plots\{file}.png')
    choice = yes_or_no('done, continue?')
    if choice in ['1', 'y', '']:
        pass
    else:
        break
