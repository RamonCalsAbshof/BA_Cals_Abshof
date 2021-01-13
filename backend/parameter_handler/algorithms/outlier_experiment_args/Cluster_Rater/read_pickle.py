import pickle
import numpy as np
import pandas
import matplotlib.pyplot as plt


def read_pickle():
    path = '/home/tatusch/Dokumente/Forschung/KI-Projekt/data_wesad/WESAD/'

    for i in range(2, 18):
        if i != 12:
            file = open(path+'S'+str(i)+'/'+'S'+str(i)+'.pkl', 'rb')
            u = pickle._Unpickler(file)
            u.encoding = 'latin1'
            p = u.load()
            keys = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
            for key in keys:
                if key == 'ACC':
                    data = np.transpose(p['signal']['chest'][key])
                else:
                    data = np.append(data, [p['signal']['chest'][key].flatten()], axis=0)

            data = np.append(data, [p['label'].flatten()], axis=0)
            data = np.transpose(data)
            df = pandas.DataFrame(data, columns=['ACC1', 'ACC2', 'ACC3', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp', 'State'])
            df.to_csv('WESAD_'+'S'+str(i)+'.csv', index=False)
            print(i)


def plot_frames():
    ax = plt.gca()
    for i in range(2, 18):
        if i != 12:
            df = pandas.read_csv('WESAD_'+'S'+str(i)+'.csv')
            df['index1'] = df.index
            # df2 = df.head(1500)
            # df2.tail(500).plot(kind='line', x='index1', y='EDA', ax=ax)
            df.head(1500).tail(500).plot(kind='line', x='index1', y='ACC3', ax=ax)
    plt.show()


def gen_pakdd_test_file():
    frames = []
    for i in range(2, 18):
        if i != 12:
            df = pandas.read_csv('WESAD_' + 'S' + str(i) + '.csv')
            df['time'] = df.index
            df['object_id'] = i
            df = df.head(1600)
            df = df.tail(100)
            frames.append(df)
    result_frame = pandas.concat(frames)
    result_frame.to_csv('WESAD_PAKDD.csv', index=False)


gen_pakdd_test_file()