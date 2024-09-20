import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp


E1 = "C:/Users/doyle/OneDrive/Desktop/Programming/Python/School/PHY245/Lab_1/Data/Exercise 1.csv"


def find_trials(data_frame):
    """
    Combs through the headers and returns a list of all the columns
     that start with that name!
    :param data_frame:
    :return:
    """
    trial_start_name = "Time"

    starts = []
    for i, head in enumerate(data_frame.columns):
        if head.split()[0].lower() == trial_start_name.lower():
            starts.append(i)

    starts.append(len(data_frame.columns))

    return starts



def create_dataframes(data_frame, index_list):
    """
    Creates the dataframes for each trial in index_list!
    Pulls out columns which start with the names given

    Must go time, position (angle), velocity
    :param data_frame:
    :param index_list:
    :return:
    """
    # new names is the map from column_names to new_names where
    # every name in column_names becomes the name in new_names
    column_names = ['Time (s)', 'Angle (rad)', 'Angular Velocity']
    new_names = ['Time', 'Angle', 'Angular_Velocity']

    # generate a list of smaller dataframes containing each trial
    trials = []
    for f in range(len(index_list) - 1):
        start_i = index_list[f]
        end_i = index_list[f+1]
        new_df = data_frame.iloc[:, start_i:end_i]
        trials.append(new_df)


    data_frames = []
    for trial in trials:

        # name_dict holds the mapping from the first two words in the name (given by column_names)
        # to the actual column name in the trial's dataframe.
        name_dict = {}
        for i, c_name in enumerate(column_names):
            for name in trial.columns:
                if c_name.split()[:2] == name.split()[:2]:
                    name_dict[new_names[i]] = name

        # value_dict maps the name in new_names to the data held by the trial's dataframe
        value_dict = dict()
        for name in name_dict:
            value_dict[name] = trial[name_dict[name]]

        # create a dataframe with the dictionary mapping the column name to the values held.
        new = pd.DataFrame(value_dict)
        data_frames.append(new)

    return data_frames



def all_to_numpy(data_frames):
    """
    Converts the list of dataframes to numpy arrays

    Goes time, position, velocity
    :param data_frames:
    :return:
    """
    np_arrays = []
    for df in data_frames:
        new = df.to_numpy()
        np_arrays.append(new)


    return np_arrays



def find_peaks(data_frame):
    """
    Finds the peaks in the given data_frame!

    Adds to the 4th row whether this is a maximum (1) or minimum (0) value
    :param data_frame:
    :return:
    """
    # columns = ['Time', 'Angle', 'Angular_Velocity']
    # finds when angular_velocity is 0! (within margin of delta)

    # dist ensures peaks are far enough away.
    # This way, we're not selecting multiple points for the same peak!
    dist = 20

    # size checks that it is a peak relative to this many datapoints away.
    # Noise might be a local maximum relative to the nearest 2 datapoints, but
    # is actually part of a local minimum when considering the nearest 20 points!
    # that they are peaks rel
    size = 20

    pos = data_frame[:, 1]
    neg = data_frame[:, 1] * -1

    peaks = sp.signal.find_peaks(pos, distance=dist, width=size)
    valleys = sp.signal.find_peaks(neg, distance=dist, width=size)

    indices_1 = peaks[0]
    indices_2 = valleys[0]

    zero_values_1 = np.c_[data_frame[indices_1], np.ones((indices_1.shape[0], 1))]
    zero_values_2 = np.c_[data_frame[indices_2], np.zeros((indices_2.shape[0], 1))]

    zero_values = np.concatenate((zero_values_1, zero_values_2))

    return zero_values



def calc_periods(max_min_rows):
    """
    Calculates the periods for the trials!

    Returns an array with combined periods of [height, period]

    :param max_min_rows:
    :return:
    """
    max = max_min_rows[np.where(max_min_rows[:, 3] == 1)]
    min = max_min_rows[np.where(max_min_rows[:, 3] == 0)]

    max_periods = []
    for p_index in range(max.shape[0] - 1):
        # period is distance between two peaks
        # value is the height of that peak!
        period = max[p_index + 1, 0] - max[p_index, 0]
        value = max[p_index, 0]
        max_periods.append([value, period])

    max = np.array(max_periods)

    min_periods = []
    for p_index in range(min.shape[0] - 1):
        # period is distance between two peaks
        # value is the height of that peak!
        period = min[p_index + 1, 0] - min[p_index, 0]
        value = min[p_index, 0]
        min_periods.append([value, period])

    min = np.array(max_periods)

    total = np.concatenate((max, min))
    return total


def lin_reg_on_param(parameter, data):
    """
    Takes an array of parameters and an array of data.
    Then performs a linear regression on the data based on the parameter.

    Initially used to create linear regressions from the average period against the length of the pendulum.
    :param parameter:
    :param data:
    :return:
    """
    res = sp.stats.linregress(parameter, data)
    return res





# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    new_row_name = "time"

    raw_data = pd.read_csv('Data/Exercise 1.csv')

    starts = find_trials(raw_data)
    trial_frames = create_dataframes(raw_data, starts)

    t1 = all_to_numpy(trial_frames)
    peaks = find_peaks(t1[0])

    all_periods = []
    all_means = []
    all_std = []
    datapoints = np.zeros((len(t1), 2))

    for i, entry in enumerate(t1):
        peaks = find_peaks(entry)
        periods = calc_periods(peaks)
        all_periods.append(periods)
        # print(f'mean: {np.mean(periods, axis=0)}')
        mean = np.mean(periods, axis=0)
        std = np.std(periods, axis=0)

        datapoints[i, :] = [mean[1], std[1]]

    # print(f'datapoints: {datapoints}')
    # print(f'datapoints: {datapoints[:, 0]}')


    params = np.array([30.9, 30.9, 30.9, 30.1, 30.1, 30.1, 28.2, 28.2, 28.2, 26.4, 26.4, 26.4, 23.2, 23.2, 23.2, 19.2, 19.2, 19.2, 16.5, 16.5, 16.5, 12.2, 12.2, 12.2, 9.6, 9.6, 9.6, 33.9, 33.9, 33.9])
    params = (params * 75.7 + 27.4 * 27.8) / (75.7 + 27.8)

    params = np.divide(params, 100)
    print(f'params: {params}')
    params = np.sqrt(params)
    print(f'params: {params}')
    reg_line = lin_reg_on_param(params, datapoints[:, 0])


    # obtain the means and standard deviations from datapoints array
    all_means = datapoints[:, 0]
    all_std = datapoints[:, 1]

    # print(f'all_means and all_std:')
    # for row in datapoints:
    #     print(f'{row[0]}')
    #
    # for row in all_means:
    #     print(row)


    f, ax = plt.subplots(1, 1, figsize=(10, 10))


    # ------ Position and Velocity -----
    print(t1[0][:, 1])
    position = t1[0][:, 1]
    velocity = t1[0][:, 2]
    print(t1[0])
    sns.scatterplot(ax=ax, x=position, y=velocity, label='length=0.309m')
    # sns.histplot(x=position, y=velocity, stat='density', bins=20, pthresh=.0000001, cmap="mako")
    ax.set(xlabel='Angular Position (rad)', ylabel='Angular Velocity (rad/sec)', title='Position against velocity at 0.309m')




    # ------ Linear Regression -----

    # # plot datapoints with error bars
    # sns.scatterplot(ax=ax, x=params, y=all_means, label='original_data')
    # plt.errorbar(x=params, y=all_means, yerr=all_std, xerr=(0.0005 ** 0.5), fmt='.', color='blue', ecolor='lightgray')
    #
    #
    # # add little text with description of linear regression
    # add_string = f'y = {reg_line.slope :.4f} * x + {reg_line.intercept :.4f}\n'
    # add_string += f'r^2 = {reg_line.rvalue :.4f}\n'
    # add_string += f'stderr = {reg_line.stderr :.6f}'
    # ax.text(0.375, 1.05, add_string, fontsize=10)
    #
    # # create values to plot linear regression line
    # x_vals = np.array([np.amax(params), np.amin(params)])
    # y_vals = reg_line.intercept + reg_line.slope * x_vals
    #
    # # plot linear regression line
    # sns.lineplot(ax=ax, x=x_vals, y=y_vals, color="r", label=f'linear regression:\n{reg_line.slope :.4f} * x + {reg_line.intercept :.4f}')
    #
    # # label axis
    # ax.set(xlabel='Sqrt of Length of Pendulum (m^1/2)', ylabel='Mean Period (sec)', title='Mean Period Against Length of Pendulum')


    plt.legend()
    plt.show()




