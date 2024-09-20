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





# plotDF = pd.DataFrame(data=trial, columns=["Time (s) E2-L1-319-T1", "Angle (rad) E2-L1-319-T1"])

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
    for entry in t1:
        peaks = find_peaks(entry)
        periods = calc_periods(peaks)
        all_periods.append(periods)
        print(f'mean: {np.mean(periods, axis=0)}')
        mean = np.mean(periods, axis=0)
        all_means.append(mean[1])

    print(f'all means: {all_means}')
    # periods = calc_periods(peaks)





    plotDF = trial_frames[0]
    # peaks = find_peaks(plotDF)


    # plotDF = pd.DataFrame(data=trial, columns=["Time (s) E2-L1-319-T1", "Angle (rad) E2-L1-319-T1"])

    # sns.set_color_codes("pastel")
    f, ax = plt.subplots(3, 1, figsize=(20, 10))

    # ax = plt.subplot(1, 2, 1)

    # sns.scatterplot(ax=ax[0], data=trial_frames[0], x="Time", y="Angular_Velocity")
    # sns.scatterplot(ax=ax[1], data=trial_frames[1], x="Time", y="Angular_Velocity")
    # sns.scatterplot(ax=ax[2], data=trial_frames[2], x="Time", y="Angular_Velocity")


    # sns.scatterplot(data=trial_frames[0], x="Time", y="Angle")
    #
    # # zero = peaks[0]
    # for zero in peaks:
    #     plt.scatter(x=zero['Time'], y=zero['Angle'], color='r')
    i = 1

    for c in range(3):
        index = i * 3 + c
        peaks = find_peaks(t1[index])
        periods = calc_periods(peaks)
        all_periods.append(periods)
        sns.scatterplot(ax=ax[c], x=t1[index][:, 0], y=t1[index][:, 1])
        sns.scatterplot(ax=ax[c], x=peaks[:, 0], y=peaks[:, 1], hue=peaks[:, 3], palette="ch:r=-.5,l=.75")


    # sns.scatterplot(x=t1[0][:, 0], y=t1[0][:, 1])
    # sns.scatterplot(x=peaks[:, 0], y=peaks[:, 1], hue=peaks[:, 3], palette="ch:r=-.5,l=.75")



    # for i in range(3):
    #     sns.scatterplot(ax=ax[i], data=trial_frames[i], x="Time", y="Angular_Velocity")
    #     peaks = find_peaks(trial_frames[i])
    #     print(peaks.columns)
    #
    #     sns.scatterplot(ax=ax[i], data=peaks, x='Time', y='Angular_Velocity', color='r')




        # for zero in peaks:
        #     sns.scatterplot(ax=ax[i], x=zero['Time'], y=zero['Angle'], color='r')


    # ax[0, 0].set_title("test 1")

    plt.show()
    # plt.figure()

    # trial[0].plot(x='Time (s) E2-L1-319-T1 (4055, 7)', y='Acceleration (m/s²) E2-L1-319-T1')



