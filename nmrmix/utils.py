import numpy as np


def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def create_interval_list(
    shifts_pred_mu,
    ppm_span=1,
    pad=None,
):
    intervals = []
    for molecule in shifts_pred_mu:
        temp = []
        for shift in molecule:
            print("shift", shift)
            if shift != pad:
                temp.append([shift - ppm_span, shift + ppm_span])
        print("testing temp", temp)
        temp = merge_intervals(temp)
        intervals.append(temp)
    return intervals


def merge_all_intervals(shifts_pred_mu, ppm_span=1, pad=None):
    interval_matrix = create_interval_list(shifts_pred_mu, ppm_span, pad)
    intervals = []
    for idx, row in enumerate(interval_matrix):
        for interval in row:
            intervals.append([interval, [idx]])

    print(intervals)
    intervals.sort(key=lambda x: x[0][0])

    merged = []
    for interval in intervals:
        # print(interval)
        if not merged or merged[-1][0][1] < interval[0][0]:
            merged.append(interval)
        else:
            merged[-1][0][1] = max(merged[-1][0][1], interval[0][1])
            merged[-1][1] = np.concatenate([merged[-1][1], interval[1]])
    return [i for i in reversed(merged)]
