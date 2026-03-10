import os
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)


DATA_PATH = "/content/drive/MyDrive/Data/New_VitalDB_2peaks_Onset.pkl"
VAL_DATA_PATH = "/content/drive/MyDrive/Test and validation data/Validation/VitalDB_Onset_val.pkl"

MODEL_PATH = "/content/drive/MyDrive/customized_normalized_onsets_PPG_models/best_PPG_onset_customized_normalized_non_quantized.keras"
OUTPUT_DIR = "/content/drive/MyDrive/customized_normalized_onsets_PPG_models"

RESULT_PATH = "/content/drive/MyDrive/customized_normalized_onsets_PPG_models/tflite_result.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_dataset(df):

    indices = df.index

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42
    )

    df_train = df.loc[train_idx]
    df_val = df.loc[val_idx]
    df_test = df.loc[test_idx]

    return df_train, df_val, df_test


def load_signal(df):

    X = np.stack(df["PPG_Signal_Normalized"].values)
    y = df["Result"].values

    return X, y
def representative_dataset_gen(rep_data):

    for x in rep_data:

        x = x.astype(np.float32)
        x = x.reshape(1, 98, 1)

        yield [x]


def convert_model(model_path, rep_data,
                  use_new_quantizer=True,
                  use_new_converter=True,
                  int8_mode="strict"):

    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = lambda: representative_dataset_gen(rep_data)

    converter._experimental_new_quantizer = use_new_quantizer
    converter.experimental_new_converter = use_new_converter

    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    if int8_mode == "strict":

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]

        converter.target_spec.supported_types = [tf.int8]

    else:

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]

        converter.target_spec.supported_types = [tf.int8]

    try:

        tflite_model = converter.convert()

        return tflite_model, None

    except Exception as e:

        return None, str(e)


def clarke_error_grid_zone(y_true, y_pred):

    if y_true < 70 and y_pred < 70:
        return "A"

    elif 0.8 * y_true <= y_pred <= 1.2 * y_true:
        return "A"

    elif (y_true >= 70 and y_true <= 290 and y_pred > y_true + 110) or \
         (y_true >= 130 and y_true <= 180 and y_pred < (7/5) * y_true - 182):

        return "C"

    elif (y_true > 240 and 70 <= y_pred <= 180) or \
         (y_true < 58 and 70 <= y_pred <= 180) or \
         (y_true >= 58 and y_true <= 70 and y_pred > 1.2 * y_true):

        return "D"

    elif (y_true > 180 and y_pred < 70) or (y_true < 70 and y_pred > 180):

        return "E"

    else:
        return "B"
    
def predict(model_path, X_data):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    preds = []
    for i in range(len(X_data)):
        sample = X_data[i].reshape(1, X_data.shape[1], 1).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])
        preds.append(pred.flatten()[0])
    return np.array(preds)



def evaluate_model(model_path, X_test, y_test):

    pred = predict(model_path, X_test)

    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    mard = np.mean(np.abs((y_test - pred) / y_test)) * 100

    zones = [
        clarke_error_grid_zone(y_t, y_p)
        for y_t, y_p in zip(y_test, pred)
    ]

    zone_counts = pd.Series(zones).value_counts(normalize=True) * 100

    results = [{
        "MSE": mse,
        "RMSE": rmse,
        "MARD": mard,
        "Zone_A": zone_counts.get("A", 0),
        "Zone_B": zone_counts.get("B", 0),
        "Zone_C": zone_counts.get("C", 0),
        "Zone_D": zone_counts.get("D", 0),
        "Zone_E": zone_counts.get("E", 0)
    }]

    return results[0]



def main():

    df = pd.read_pickle(DATA_PATH)

    df_train, df_val, df_test = split_dataset(df)

    X_val, y_val = load_signal(df_val)
    X_test, y_test = load_signal(df_test)

    rep_sizes = [2000, 3000, 4000, 5000]

    base_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

    all_results = []

    for rep_size in rep_sizes:

        rep_data = X_val[:rep_size]

        out_name = f"{base_name}_rep{rep_size}.tflite"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if os.path.exists(out_path):

            print("[SKIP]", out_name, "already exists")

            all_results.append({
                "model": out_name,
                "rep_size": rep_size,
                "status": "SKIPPED"
            })

            continue

        print("Converting with rep size:", rep_size)

        tflite_model, error = convert_model(
            MODEL_PATH,
            rep_data
        )

        if tflite_model is not None:

            with open(out_path, "wb") as f:
                f.write(tflite_model)

            status = "SUCCESS"
            val_metrics = evaluate_model(out_path, X_val, y_val)
            test_metrics = evaluate_model(out_path, X_test, y_test)

            all_results.append({
                "model": out_name,
                "rep_size": rep_size,
                "status": status,
                **{f"val_{k}": v for k,v in val_metrics.items()},
                **{f"test_{k}": v for k,v in test_metrics.items()}
            })

        else:

            status = f"FAILED: {error}"
            all_results.append({
                "model": out_name,
                "rep_size": rep_size,
                "status": status
            })

        print(out_name, "→", status)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULT_PATH, index=False)


if __name__ == "__main__":
    main()