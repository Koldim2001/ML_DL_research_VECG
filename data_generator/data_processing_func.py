from func import *
import numpy as np


def filter_by_acp_presence(signal, threshold):
    sharp_peaks = check_if_acp(signal, threshold)
    #print(sharp_peaks)
    if len(sharp_peaks) >= threshold:
        return True
    return False


def check_if_acp(signal, threshold):
    sharp_peaks = []
    for i in range(1, len(signal)-1):
        diffs = (np.asarray(signal[i-2:i+2]) - signal[i]) / signal[i] * (-1)
        diff_high = np.where(diffs > 0.9)[0]
        if len(diff_high) >= 2 and np.all(diffs >= 0) and signal[i] > np.std(signal): 
            sharp_peaks.append(i)
    return sharp_peaks


def get_qrs_widths(peaks, time):
    complexes = []
    widths = []
    k = 0
    last_s = -1
    while k < len(peaks['ECG_Q_Peaks']) and k < len(peaks['ECG_S_Peaks']):
        first_q_peak = peaks['ECG_Q_Peaks'][k]
        if first_q_peak > last_s: 
            first_s_peak = -1    
            for i in range(len(peaks['ECG_S_Peaks'])):
                if peaks['ECG_S_Peaks'][i] > first_q_peak:
                    first_s_peak = peaks['ECG_S_Peaks'][i]
                    break
            if first_s_peak != -1:
                complexes.append([first_q_peak, first_s_peak])
                widths.append(time[first_s_peak] - time[first_q_peak])
        k += 1
    return complexes, widths


def filter_by_qrs_complexes_widths(peaks, time, width_threshold=0.12):
    complexes, widths = get_qrs_widths(peaks, time)
    if np.mean(widths) > width_threshold:
        return True
    return False


def filter_by_different_qrs_form(signal, peaks, time, diff_threshold=0.95):
    complexes, widths = get_qrs_widths(peaks, time)
    correlations = get_qrs_form_diff(signal, complexes)
    unsimilar_qrs = np.where(correlations < diff_threshold)
    if len(unsimilar_qrs[0])/len(correlations) > 0.75:
        return True
    return False


def get_qrs_form_diff(signal, complexes):
    corrs = []
    for i in range(len(complexes) - 1):
        for j in range(i+1, len(complexes)):
            qrs1 = signal[complexes[i][0]:complexes[i][1]]
            qrs1 = qrs1/np.max(np.abs(qrs1))
            qrs2 = signal[complexes[j][0]:complexes[j][1]]
            qrs2 = qrs2/np.max(np.abs(qrs2))
            c11 = np.correlate(qrs1, qrs1, mode='full')
            c22 = np.correlate(qrs2, qrs2, mode='full')
            c12 = np.correlate(qrs1, qrs2, mode='full')
            corrs.append(np.max(c12)/ max(np.max(c11), np.max(c22)))
    return np.asarray(corrs)

