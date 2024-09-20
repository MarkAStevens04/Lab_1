import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp


E1 = "C:/Users/doyle/OneDrive/Desktop/Programming/Python/School/PHY245/Lab_1/Data/Exercise 1.csv"


entry_names = []


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
        entry_names.append(new_df.columns)


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
    Converts the list of dataframes to numpy arrays.
    Also removes columns which are all NaN

    Goes time, position, velocity
    :param data_frames:
    :return:
    """
    np_arrays = []
    for df in data_frames:
        new = df.to_numpy()
        # remove rows from dataset if every entry is NaN

        # extract full array but with NaN replaced with True and else replaced with False
        a = np.isnan(new)
        # create 1-D slide of array
        # True if every entry in original row was True
        # False if at least one entry was False
        # Should be Nx1 array, with N=number datapoints
        b = ~np.all(a, axis=1)
        # C is our original array indexed at these new values
        c = new[b]
        np_arrays.append(c)


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
    dist = 3

    # size checks that it is a peak relative to this many datapoints away.
    # Noise might be a local maximum relative to the nearest 2 datapoints, but
    # is actually part of a local minimum when considering the nearest 20 points!
    # that they are peaks rel
    size = 2

    pos = data_frame[:, 1]
    neg = data_frame[:, 1] * -1

    # print(f'finding peaks: {data_frame}')

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


def return_entry_titles():
    """
    Return the titles of the entries!
    Uses the entry_names array
    :return:
    """
    # index where the name is located
    name_index = 0

    # number of words in title to include
    num_words = 1

    titles = []
    for headers in entry_names:
        # get the header
        header_title_full = headers[name_index]
        # extract the last word of the full header title
        header_title_split = header_title_full.split()[-1 * num_words:]
        # extract that phrase
        header_title = ''
        for h in header_title_split:
            header_title += h
            header_title += ' '
        header_title = header_title[:-1]
        titles.append(header_title)
        print(header_title)
    return titles

def extract_errors(params):
    """
    Takes a matrix [[entry 1, entry 1 + error, entry 1 - error], [...], ...]
    and extracts the error!
    Returns:

    [entry 1, entry 2, entry 3, ...], [[entry1 + error, entry2 + error, ...], [entry1 - error, entry2 - error, ...]]

    :return:
    """
    errors = params[:, -2:]
    errors[:, 0] = params[:, 0] - params[:, 1]
    errors[:, 1] = params[:, 0] - params[:, 2]
    errors = np.abs(errors)
    errors = np.transpose(errors)
    print(f'errors: {errors}')
    params = params[:, 0]
    return params, errors


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    new_row_name = "time"

    raw_data = pd.read_csv('Data/Exercise 5.csv')

    starts = find_trials(raw_data)
    trial_frames = create_dataframes(raw_data, starts)

    # print(f'found frames: {trial_frames}')

    t1 = all_to_numpy(trial_frames)
    # print(f'turned to numpy: {t1}')
    peaks = find_peaks(t1[0])
    # print(f'peaks: {peaks}')



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
    print(f'datapoints: {datapoints}')

    # print(f'datapoints: {datapoints}')
    # print(f'datapoints: {datapoints[:, 0]}')


    # params = np.array([16.2, 16.2, 16.2, 16.2, 16.2, 16.2, 16.3677095, 16.3677095, 16.3677095, 16.49453616, 16.49453616, 16.49453616, 16.47455317, 16.47455317, 16.47455317])
    params = np.array([27.8, 27.8, 27.8, 103.5, 103.5, 103.5, 179, 179, 179, 254.4, 254.4, 254.4, 330.1, 330.1, 330.1])

    # For exercise 4, should do errors-in-variables regression
    p_error = 0.1

    # carries error through transformations
    params = np.c_[params, params + [0.5] * params.shape[0], params - [0.5] * params.shape[0]]

    # change params to plot
    # params = params / 1000

    # extract errors from params
    # if you want a constant error, delete the line that concatenates the error and the next line, and set the errors to a constant value.
    params, errors = extract_errors(params)


    # perform our linear regression
    reg_line = lin_reg_on_param(params, datapoints[:, 0])

    # obtain the means and standard deviations from datapoints array
    all_means = datapoints[:, 0]
    all_std = datapoints[:, 1]

    print(f'all_means and all_std:')
    for row in all_means:
        print(row)
    # for row in all_std:
    #     print(row)


    f, ax = plt.subplots(1, 1, figsize=(10, 10))

    # ------ Debug Peak Finding -----

    # # num trials per segment
    # # If you do 3 trials at 30.9m, your trials per segment is 3
    # tps = 3
    # i = 2
    #
    # f, ax = plt.subplots(tps, 1, figsize=(20, 10))
    # e_titles = return_entry_titles()
    #
    #
    #
    # for c in range(tps):
    #     index = i * tps + c
    #     peaks = find_peaks(t1[index])
    #     periods = calc_periods(peaks)
    #     sns.scatterplot(ax=ax[c], x=t1[index][:, 0], y=t1[index][:, 1])
    #     sns.scatterplot(ax=ax[c], x=peaks[:, 0], y=peaks[:, 1], hue=peaks[:, 3], palette="ch:r=-.5,l=.75")
    #     ax[c].set(title=e_titles[index])


    # ------ Position and Velocity -----

    # position = t1[0][:, 1]
    # velocity = t1[0][:, 2]
    # sns.scatterplot(ax=ax, x=position, y=velocity, label='length=0.309m')
    # # sns.histplot(x=position, y=velocity, stat='density', bins=20, pthresh=.0000001, cmap="mako")
    # ax.set(xlabel='Angular Position (rad)', ylabel='Angular Velocity (rad/sec)', title='Position against velocity at 0.309m')



    # ------ Linear Regression -----

    # plot datapoints with error bars
    sns.scatterplot(ax=ax, x=params, y=all_means, label='original_data')
    plt.errorbar(x=params, y=all_means, yerr=all_std, xerr=errors, fmt='.', color='blue', ecolor='lightgray')


    # add little text with description of linear regression
    add_string = f'y = {reg_line.slope :.4f} * x + {reg_line.intercept :.4f}\n'
    add_string += f'$r^2$ = {reg_line.rvalue :.4f}\n'
    add_string += f'stderr = {reg_line.stderr :.6f}'

    # x and y are positions on graph where top left corner should be.
    # x and y values are values of each variable (like 0.1 radians or something)
    ax.text(25, 0.7, add_string, fontsize=10)

    # create values to plot linear regression line
    x_vals = np.array([np.amax(params), np.amin(params)])
    y_vals = reg_line.intercept + reg_line.slope * x_vals

    # plot linear regression line
    sns.lineplot(ax=ax, x=x_vals, y=y_vals, color="r", label=f'linear regression:\n{reg_line.slope :.4f} * x + {reg_line.intercept :.4f}')

    # label axis
    ax.set(xlabel='Total weight of oscillator (g)', ylabel='Mean Period (sec)', title='Mean Period Against Weight')

    plt.ylim(0)
    plt.legend()
    plt.show()




