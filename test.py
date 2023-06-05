import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


#%% Plot a specific anomaly
# station = 905
# # Read the database
# df = pd.read_csv(f'data/labeled_{station}_cle.csv', sep=',', encoding='utf-8', parse_dates=['date'])

# # Normalize the data
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# df.iloc[:, 1:-1] = scaler.fit_transform(df.iloc[:, 1:-1])

# # Set date as the index column
# df.set_index('date', inplace=True)

# # Drop the label column
# df.drop(df.columns[-1], axis=1, inplace=True)

# # deletion: 330405:330572
# # Rest = 451872:452041
# fig = df.iloc[451872:452041, :].plot(figsize=(10,5))
# plt.title(f"Anomaly 50 station {station}")
# plt.xlabel('Date')
# plt.ylabel('Standarized values')
# # plt.show()

# # Save the image
# fig = fig.get_figure()
# fig.subplots_adjust(bottom=0.19)
# fig.savefig(f'images/anomaly_{station}_50_trees.png', dpi=300)

#%% Plot the labels of a specific anomaly
station = 905

# Read the database
df = pd.read_csv(f'data/labeled_{station}.csv', sep=',', encoding='utf-8', parse_dates=['date'])

# Load the labels and the predictions
y = df['label'].to_list()
y_logreg = np.load('y_logreg.npy', allow_pickle=True)
y_amf = np.load('y_amf.npy', allow_pickle=True)
y_arf = np.load('y_arf.npy', allow_pickle=True)

# Reformat the data
y_logreg = [0 if i[False] >= 0.5 else 1 for i in y_logreg]
y_amf = [0 if (len(i)==0) or (i[0] >= 0.5) else 1 for i in y_amf]
y_arf = [0 if (len(i)==0) or (i[0] >= 0.5) else 1 for i in y_arf]

# Anomaly 43 = 446208:446305
# Anomaly 50 = 451872:452041
stretch = 36
start_index = 446208 - stretch
end_index = 446305 + stretch
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y[start_index:end_index], label='label')
ax.plot(y_logreg[start_index:end_index], label='Log. Reg.')
ax.plot(y_amf[start_index:end_index], label='AMF')
ax.plot(y_arf[start_index:end_index], label='ARF')
ax.set_title('Anomaly 43 station 905')
ax.set_xlabel('Time')
ax.set_ylabel('Label')
plt.legend()
plt.show()

# Add title and axis labels

