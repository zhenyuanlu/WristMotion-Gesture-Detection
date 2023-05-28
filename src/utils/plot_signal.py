from utils import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

column_names = ['timestamp', 'channel_0_raw', 'channel_1_raw', 'channel_0_high_passed', 'channel_1_high_passed',
                'quaternion_x', 'quaternion_y', 'quaternion_z', 'quaternion_w',
                'gyroscope_x', 'gyroscope_y', 'gyroscope_z',
                'acc_x', 'acc_y', 'acc_z',
                'body_movement', 'repetition_number']

df = pd.read_csv(r'Z:\Pison\pison_movement\data\pison.csv', names=column_names)

labels = [0, 1, 2, 3, 4]
reps = [1, 2, 3]

groups = df.groupby(df.columns[-1])

# Create a subplot for each group
fig, axs = plt.subplots(len(groups), 1, sharex=True, figsize=(10, 10))
selected = 10
# Plot each group
for ax, (name, group) in zip(axs, groups):
    for subname, subgroup in group.groupby(df.columns[-2]):
        ax.plot(subgroup[df.columns[selected]].values, label=f'Class {name}- Movement {subname}')
    ax.legend()
    ax.set_title(f'Feature: {column_names[selected]}')


plt.tight_layout()
plt.show()


