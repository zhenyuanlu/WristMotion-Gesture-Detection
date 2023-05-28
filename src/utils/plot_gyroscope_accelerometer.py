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

# groups = df.groupby(df.columns[-1])
#
# # Create a subplot for each group
# fig, axs = plt.subplots(len(groups), 1, sharex=True, figsize=(10, 10))
# selected = 2
# # Plot each group
# for ax, (name, group) in zip(axs, groups):
#     for subname, subgroup in group.groupby(df.columns[-2]):
#         ax.plot(subgroup[df.columns[selected]].values, label=f'Class {name}- Movement {subname}')
#     ax.legend()
#     ax.set_title(f'Feature: {column_names[selected]}')
#
#
# plt.tight_layout()
# plt.show()




# fig = plt.figure(figsize=(10, 10))
#
# # Loop over each group
# for i, (name, group) in enumerate(groups):
#     ax = fig.add_subplot(len(groups), 1, i+1, projection='3d')
#
#     # Loop over each subgroup within the group
#     for subname, subgroup in group.groupby(df.columns[-2]):
#         ax.plot(subgroup['quaternion_x'], subgroup['quaternion_y'], subgroup['quaternion_z'], label=f'{name}-{subname}')
#
#     ax.set_xlabel('X Acceleration')
#     ax.set_ylabel('Y Acceleration')
#     ax.set_zlabel('Z Acceleration')
#     ax.legend()
#     ax.set_title(f'Group: {name}')
#
# plt.tight_layout()
# plt.show()
#

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
#
# gyroscope  = df
#
# gyroscope  = gyroscope [(gyroscope ['body_movement']==0) & (gyroscope ['repetition_number'] ==2)]
# gyroscope  = gyroscope .iloc[:, 9:12]
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(gyroscope['gyroscope_x'], gyroscope['gyroscope_y'], gyroscope['gyroscope_z'])
#
# # Function to update plot for each frame
# def update(frame):
#     ax.view_init(30, frame/2)
#     plt.draw()
#
# # Create animation
# ani = FuncAnimation(fig, update, frames=range(0, 360, 2), interval=100)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Assuming gyroscope and accelerometer data are pandas dataframes with columns 'x', 'y', 'z'
gyroscope = df[(df['body_movement']==0) & (df['repetition_number'] ==2)].iloc[:, 9:12]
accelerometer = df[(df['body_movement']==0) & (df['repetition_number'] ==2)].iloc[:, 12:15]

# Plot the gyroscope data
ax.scatter(gyroscope['gyroscope_x'], gyroscope['gyroscope_y'], gyroscope['gyroscope_z'])

# Add arrows for the accelerometer data
ax.quiver(accelerometer['acc_x'], accelerometer['acc_y'], accelerometer['acc_z'],
          np.sin(np.pi * accelerometer['acc_x']),
          -np.cos(np.pi * accelerometer['acc_y']),
          np.sqrt(2.0 / 3.0) * np.cos(np.pi * accelerometer['acc_z']),
          length=0.1, normalize=True)

plt.show()
