import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm.keras import TqdmCallback
import argparse
sns.set_theme(style="whitegrid")

def load_data(path, signal_type):
    df = pd.read_pickle(path)
    if signal_type == "PPG":
        X = np.stack(df["PPG_Signal_Normalized"].values)
    elif signal_type == "VPG":
        X = np.stack(df["VPG_Signal_Normalized"].values)
    elif signal_type == "APG":
        X = np.stack(df["APG_Signal_Normalized"].values)
    y = df["Result"].values
    return X, y

def split_dataset(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def Conv_1D_Block(x, filters, kernel_size=3, strides=1):
    x = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def Residual_Block(x, filters, downsample=False):
    shortcut = x
    if downsample or x.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, strides=2 if downsample else 1, padding="same", kernel_initializer="he_normal")(x)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Conv1D(filters, kernel_size=3, strides=2 if downsample else 1, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv1D(filters, kernel_size=3, padding="same", kernel_initializer="he_normal")(x)
    # x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x

def MLP(x, size_list, dropout_rate, output_nums, problem_type):
    x = tf.keras.layers.Flatten(name='flatten')(x)

    for i, size in enumerate(size_list):
        x = tf.keras.layers.Dense(size, activation='relu', name=f'dense_{size}')(x)
        x = tf.keras.layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)

    activation = 'softmax' if problem_type == 'Classification' else 'linear'
    outputs = tf.keras.layers.Dense(output_nums, activation=activation, name="output")(x)

    return outputs

def build_resnet34(
    length,
    num_filters_residual,
    num_layers,
    final_layer_size,
    dropout_rate,
    num_channel=1,
    num_filters=64,
    output_nums=1,
    problem_type="Regression"
):

    inputs = tf.keras.Input((length, num_channel))

    x = Conv_1D_Block(inputs, num_filters, kernel_size=3, strides=1)

    for stage_idx, (filters, layers) in enumerate(zip(num_filters_residual, num_layers)):

        for block_idx in range(layers):

            downsample = (stage_idx > 0 and block_idx == 0)

            x = Residual_Block(x, filters, downsample=downsample)

    outputs = MLP(
        x,
        final_layer_size,
        dropout_rate,
        output_nums,
        problem_type
    )

    model = tf.keras.Model(inputs, outputs)

    return model

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_layers", type=str, default="3,4,6,3")
    parser.add_argument("--num_filters", type=str, default="8,16,32,64")
    parser.add_argument("--final_layer_size", type=str, default="256,128")
    parser.add_argument("--signal_type",
        type=str,
        default="PPG",
        choices=["PPG", "VPG", "APG"],
    )
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()

    return args

def parse_list(arg):
    return list(map(int, arg.split(",")))

def train_model(model, model_path, lr, opt, epochs, X_train, y_train, X_val, y_val):
    if opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif opt == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min'),
        ModelCheckpoint(model_path, verbose=0, monitor='val_loss', save_best_only=True, mode='min'),
        TqdmCallback(verbose=1)  # uses notebook-style progress bar in Colab
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks
    )

    return history

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


def evaluate_model(model, results_path, X_test, y_test):
    results  = []

    pred = model.predict(X_test).flatten()

    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    mard = np.mean(np.abs((y_test - pred) / y_test)) * 100

    zones = [clarke_error_grid_zone(y_t, y_p) for y_t, y_p in zip(y_test, pred)]
    zone_counts = pd.Series(zones).value_counts(normalize=True) * 100

    results.append({
        "MSE": mse,
        "RMSE": rmse,
        "MARD": mard,
        "Zone_A": zone_counts.get("A", 0),
        "Zone_B": zone_counts.get("B", 0),
        "Zone_C": zone_counts.get("C", 0),
        "Zone_D": zone_counts.get("D", 0),
        "Zone_E": zone_counts.get("E", 0)
    })
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation results saved to {results_path}")

def main():

    args = parse_args()

    num_layers = parse_list(args.num_layers)
    num_filters_residual = parse_list(args.num_filters)
    final_layer_size = parse_list(args.final_layer_size)

    data_path = "dataset.pkl"
    model_path = "best_model.keras"
    results_path = "results.csv"

    X, y = load_data(data_path, args.signal_type)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    length = X_train.shape[1]

    model = build_resnet34(
        length=length,
        num_filters_residual=num_filters_residual,
        num_layers=num_layers,
        final_layer_size=final_layer_size,
        dropout_rate=args.dropout,
    )

    model.summary()

    train_model(
        model,
        model_path,
        args.learning_rate,
        args.optimizer,
        args.epochs,
        X_train,
        y_train,
        X_val,
        y_val
    )

    model.load_weights(model_path)

    evaluate_model(
        model,
        results_path,
        X_test,
        y_test
    )

if __name__ == "__main__":
    main()
