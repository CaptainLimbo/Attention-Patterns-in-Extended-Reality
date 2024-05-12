def merge_periods(all_extracted_periods, interval_steps_tolerance):
    i = 0
    while i < len(all_extracted_periods) - 1:
        if all_extracted_periods[i+1][0] - all_extracted_periods[i][1] < interval_steps_tolerance:
            # replace the i th and i+1 th element with a merged element
            all_extracted_periods[i] = (all_extracted_periods[i][0], all_extracted_periods[i+1][1])
            all_extracted_periods.pop(i+1)
        else:
            i += 1
    return all_extracted_periods

def discard_short_periods(all_extracted_periods, minimum_fixation_steps):
    return [
        period
        for period in all_extracted_periods
        if period[1] - period[0] >= minimum_fixation_steps
    ]


def find_all_periods(signal, indices=[]):
    start = None
    all_periods = []
    for i in range(len(signal)-1):
        if start is None:
            if signal[i] and (len(indices)==0 or indices[i] + 1 == indices[i + 1]):
                start = i
        else: # there is a start
            if not signal[i] or (len(indices) > 0 and indices[i] + 1 != indices[i + 1]):
                all_periods.append((start, i))
                start = None
    if start is not None and (len(indices) == 0 or indices[len(signal) - 1] - 1 == indices[len(signal)-2]):
        all_periods.append((start, len(signal) - 1))
    return all_periods