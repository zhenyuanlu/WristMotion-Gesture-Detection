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
selected = 2
# Plot each group
for ax, (name, group) in zip(axs, groups):
    for subname, subgroup in group.groupby(df.columns[-2]):
        ax.plot(subgroup[df.columns[selected]].values, label=f'Class {name}- Movement {subname}')
    ax.legend()
    ax.set_title(f'Feature: {column_names[selected]}')


plt.tight_layout()
plt.show()




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
# gyroscope  = gyroscope [(gyroscope ['body_movement']==0) & (gyroscope ['repetition_number'] ==1)]
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


#
# # Load your data
# # Assuming quaternions is a numpy array of shape (n, 4)
# # where each row is a quaternion [x, y, z, w]
# quaternions = df
#
# quaternions = quaternions[(quaternions['body_movement']==0) & (quaternions['repetition_number'] ==1)]
# quaternions = quaternions.iloc[:, 5:9]
#
# # Normalize your quaternions if they aren't already
# norms = np.linalg.norm(quaternions, axis=1)
# quaternions = quaternions / norms[:, np.newaxis]
#
# # Convert to scipy Rotation object
rotations = R.from_quat(quaternions)
#
# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot each quaternion
# for i in range(len(quaternions)):
#     # Get the rotation matrix for this quaternion
#     matrix = rotations[i].as_matrix()
#
#     # Plot the x, y, and z vectors after rotation
#     ax.quiver(0, 0, 0, matrix[0, 0], matrix[1, 0], matrix[2, 0], color='r')  # x vector
#     ax.quiver(0, 0, 0, matrix[0, 1], matrix[1, 1], matrix[2, 1], color='g')  # y vector
#     ax.quiver(0, 0, 0, matrix[0, 2], matrix[1, 2], matrix[2, 2], color='b')  # z vector
#
# # Set plot limits
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
#
# # Show the plot
# plt.show()