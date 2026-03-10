import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, BatchNormalization, Activation, add
from keras.layers import Conv2D, MaxPooling2D, Reshape, Flatten, Dense
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
sns.set_theme(style="whitegrid")
import re
import os
from scipy.signal import butter, filtfilt, find_peaks
from scipy.signal import resample_poly
from math import gcd


def remove_outliers(signal: np.ndarray):
    """
    Return None if any abs(value) > threshold_factor * mean(abs(signal)),
    otherwise return the original signal.
    """
    abs_mean = np.mean(np.abs(signal))
    threshold = 3*abs_mean
    return signal[np.abs(signal) <= threshold]

def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 8.0,
    fs: int = 100,
    order: int = 4
) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def split_into_10s_segments(signal: np.ndarray, fs: int = 100) -> np.ndarray:
    segment_len = fs * 10 
    n_segments = len(signal) // segment_len
    return np.reshape(signal[:n_segments * segment_len], (n_segments, segment_len))

def z_score_normalize(signal: np.ndarray) -> np.ndarray:
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / std

def detect_center_peaks(
    segment: np.ndarray,
    height_thresh: float = 20,
    distance_thresh: int = 80
) -> np.ndarray:
    peaks, _ = find_peaks(segment, height=height_thresh, distance=distance_thresh)
    return peaks

def crop_1s_around_peaks(
    segment: np.ndarray,
    peaks: np.ndarray,
    window_size: int = 100
) -> List[np.ndarray]:
    half_win = window_size // 2
    cropped_segments = []
    for peak in peaks:
        start = peak - half_win
        end = peak + half_win
        if start >= 0 and end <= len(segment):
            cropped_segments.append(segment[start:end])
    return cropped_segments

def detect_peaks_with_prominence(
    ppg_segment: np.ndarray,
    height_thresh: int = 5,
    distance_thresh: int = 10,
    prominence_thresh: float = 1
) -> np.ndarray:
    peaks, _ = find_peaks(
        ppg_segment,
        height = height_thresh,
        distance=distance_thresh,
        prominence=prominence_thresh
    )
    return peaks


folder_path = r"/content/drive/MyDrive/PPG_Signal_16mins"

rows = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # Extract X (case) and Y (dt) using regex
        match = re.match(r"PPG_16mins_case_(\d+)_dt_(\d+)\.csv", filename)
        if match:
            case = int(match.group(1))
            dt = int(match.group(2))

            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath)

            ppg_signal = df['PPG_Signal'].to_numpy()
            result = df['result'].iloc[0]  

            rows.append({
                "Case": case,
                "dt": dt,
                "PPG_Signal": ppg_signal,
                "result": result
            })

df_train = pd.DataFrame(rows)
df_train['PPG_Signal_100'] = df_train['PPG_Signal'].apply(lambda x: resample_poly(x, up = 1, down = 5))
df_train['PPG_Signal_Centered'] = df_train['PPG_Signal_100'].apply(lambda x: x - np.mean(x))
df_train['PPG_Signal_Cut'] = df_train['PPG_Signal_Centered'].apply(lambda x: remove_outliers(x))
df_train['PPG_Signal_Filtered'] = df_train['PPG_Signal_Cut'].apply(lambda x: butter_bandpass_filter(x))

expanded_rows = []

for _, row in df_train.iterrows():
    signal = row['PPG_Signal_Filtered']
    case = row['case']
    dt = row['dt']
    result = row['result']

    segments = split_into_10s_segments(signal, fs=100)

    for i, segment in enumerate(segments):
        expanded_rows.append({
            'Segment_Number': i,
            'Case': case,
            'Glucose': result,
            'PPG_Segment': segment
        })
df_expanded = pd.DataFrame(expanded_rows)
print(df_expanded)

df_expanded['PPG_Signal_Reflected'] = df_expanded['PPG_Segment'].apply(lambda x: -x) #necessary to reflect because the signal is recorded in reflection mode
print (df_expanded)

df_expanded['PPG_Signal_Normalized'] = df_expanded['PPG_Signal_Reflected'].apply(
    lambda x: z_score_normalize(x)
)
print (df_train)

window_size = 100
half_win = window_size // 2
df_expanded['Peak_Constant'] = df_expanded['PPG_Segment_Reflected'].apply(lambda x: detect_center_peaks(x))
print (df_expanded)

cropped_rows = []
window_size = 100
half_win = window_size // 2

for _, row in df_expanded.iterrows():
    signal = row['PPG_Segment_Reflected']
    peak_indices = row['Peak_Constant']
    Segment_Number = row['Segment_Number']
    Case = row['Case']
    Result = row['Glucose']

    for peak in peak_indices:
        start = peak - half_win
        end = peak + half_win

        if start >= 0 and end <= len(signal):
            cropped_rows.append({
                'Peak_Index': peak,
                'Cropped_1s': signal[start:end],
                'Segment_Number': Segment_Number,
                'Case': Case,
                'Result': Result
            })
df_cropped_1s = pd.DataFrame(cropped_rows)
print(df_cropped_1s)

rows = []

for idx, row in df_cropped_1s.iterrows():
     Peak_Index = row['Peak_Index']
     seg_1s = row['Cropped_1s']
     Case = row['Case']
     Segment_Number = row['Segment_Number']
     Result = row['Result']

     peaks = detect_peaks_with_prominence(seg_1s)
     if len(peaks) >= 2:
                rows.append({
                    'Peak_Index': Peak_Index,
                    'Cropped_1s': seg_1s,
                    'Case': Case,
                    'Segment_Number': Segment_Number,
                    'Result': Result
                })

df_2peaks = pd.DataFrame(rows)
print(df_2peaks)
