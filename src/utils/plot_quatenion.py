from utils import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
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

# Load your data
# Assuming quaternions is a numpy array of shape (n, 4)
# where each row is a quaternion [x, y, z, w]
quaternions = df

quaternions = quaternions[(quaternions['body_movement']==0) & (quaternions['repetition_number'] ==1)]
quaternions = quaternions.iloc[:, 5:9]

# Normalize your quaternions if they aren't already
norms = np.linalg.norm(quaternions, axis=1)
quaternions = quaternions / norms[:, np.newaxis]

# Convert to scipy Rotation object
rotations = R.from_quat(quaternions)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each quaternion
for i in range(len(quaternions)):
    # Get the rotation matrix for this quaternion
    matrix = rotations[i].as_matrix()

    # Plot the x, y, and z vectors after rotation
    ax.quiver(0, 0, 0, matrix[0, 0], matrix[1, 0], matrix[2, 0], color='r')  # x vector
    ax.quiver(0, 0, 0, matrix[0, 1], matrix[1, 1], matrix[2, 1], color='g')  # y vector
    ax.quiver(0, 0, 0, matrix[0, 2], matrix[1, 2], matrix[2, 2], color='b')  # z vector

# Set plot limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Show the plot
plt.show()