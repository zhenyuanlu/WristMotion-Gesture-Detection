"""
process_pison.py

Preprocess pison data to npz; per npz file included all the data of one repetition
"""
import argparse
import csv
import os
import shutil
import glob
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

from utils import get_data_dir

data_dir = get_data_dir(r'data')
output_dir = get_data_dir(r'data\processed')

# timestamp (milliseconds with microsecond precision); channel 0 raw, channel 1 raw,
# channel 0 high-passed, and channel 1 high-passed (all in ADC counts); quaternion x, y, z,
# and w; gyroscope x, y, and z in degrees per second; accelerometer x, y, and z in meters per
# second squared; body movement label; and finally, repetition number (where there is one
# "repetition" for a given prompting window).

# Label dicts
pain_labels = {
    'standing #1': 0,
    'standing #2': 1,
    'walking': 2,
    'walking fast': 3,
    'running': 4
}


def _parse_args():
    """
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='process_pison.py')
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--output_dir', type=str, default=output_dir)
    parser.add_argument('--select_signal', type=int, default = 1, help='1 as GSR(EDA) signal, see colNames above')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        # Remove the existing directory
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
    return args


def standardize_length(repetitions, target_length=None, method='truncate'):
    lengths = []
    for repetition in repetitions.values():
        for chunk in repetition:
            lengths.append(len(chunk['vars']))

    if target_length is None:
        if method == 'truncate':
            target_length = 900 #min(lengths)
        elif method == 'pad':
            target_length = max(lengths)
        else:
            raise ValueError("Method must be either 'truncate' or 'pad'.")

    for repetition in repetitions.values():
        for chunk in repetition:
            if len(chunk['vars']) < target_length:
                # Padding
                pad_length = target_length - len(chunk['vars'])
                # pad with the first value
                pad_values = [chunk['vars'][0]] * pad_length
                chunk['vars'] = pad_values + chunk['vars']
            elif len(chunk['vars']) > target_length and method == 'truncate':
                # Truncating
                chunk['vars'] = chunk['vars'][-target_length:]

    return repetitions


def process_pison(data_dir, output_dir, selected_var = None):
    """
    data_dir_files contain 3 replicates for movement data.
    Each replicate has 5 labels
    :param data_dir: data directory
    :param output_dir: output directory
    :return: save pison csv files to npz
    """
    if selected_var is None:
        selected_var = list(range(1, 15))
    path = glob.glob(data_dir + '/*.csv')
    for csv_file in path:
        repetitions = defaultdict(list)
        with open(csv_file) as f:
            reader = csv.reader(f)
            # selected_data = []
            for row in reader:
                repetition_index = int(row[-2])
                label = int(row[-1]) - int(1)
                selected_values = [float(row[i]) for i in selected_var]
                if repetitions[repetition_index] and repetitions[repetition_index][-1]['label'] == label:
                    repetitions[repetition_index][-1]['vars'].append(selected_values)
                else:
                    repetitions[repetition_index].append({'vars': [selected_values], 'label': label})

            repetitions = standardize_length(repetitions, target_length=None, method='truncate')

            for rep_index, rep_list in repetitions.items():
                samples = []
                labels = []
                for rep in rep_list:
                    samples.append(rep['vars'])
                    labels.append(rep['label'])

                x = np.asarray(samples)
                # print(x.shape)
                y = np.asarray(labels)
                # print(y.shape)

                # # Scaling the data
                # scaler = MinMaxScaler()
                # x_reshaped = x.reshape(-1, x.shape[-1])
                # # print(x_reshaped.shape)
                # x_scaled = scaler.fit_transform(x_reshaped)
                # x = x_scaled.reshape(x.shape)

                print('X, y shape:{}, {}'.format(x.shape, y.shape))
                filename = os.path.join(output_dir, f'rep_{rep_index}' + '.npz')
                sample_dict = {
                    "x": x,
                    "y": y
                }
                print(sample_dict['y'])
                # print(sample_dict['x'][0])
                np.savez(filename, **sample_dict)


if __name__ == '__main__':
    args = _parse_args()
    process_pison(args.data_dir, args.output_dir)

