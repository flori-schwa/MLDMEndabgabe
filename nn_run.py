import datetime
import statistics
from typing import List

import pandas as pd
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers

import Florian
import Frank
import Phillipp
import arff
import preprocess

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe: pd.DataFrame = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


if __name__ == "__main__":
    census_raw, attributes = arff.parse_arff_file('adult_train.arff')

    tc = preprocess.TransformContext()

    if not tc.load():
        tc.calc(census_raw)
        tc.save()

    print("Transforming data...")

    tc.initialize_transformation(census_raw, attributes)

    tc.drop_attributes('fnlwgt', 'capital-loss', 'education')
    tc.transform_attribute(preprocess.class_to_target)
    tc.transform_attribute(Frank.transform_capital_gain_bin)

    attr_list = tc.transform_data('census_train', 'out/train.arff', 'out/train.csv')

    census_test, test_attr = arff.parse_arff_file("adult_test.arff")

    tc.initialize_transformation(census_test, test_attr)

    tc.transform_data('census_test', 'out/test.arff', 'out/test.csv')

    print("Running NN")

    feature_columns = []
    indicator_columns: List[arff.NominalAttribute] = []

    for attr in attr_list:
        if attr.name == "target":
            continue

        if isinstance(attr, arff.IntegerAttribute) or isinstance(attr, arff.NumericAttribute):
            feature_columns.append(feature_column.numeric_column(attr.name))
        elif isinstance(attr, arff.NominalAttribute):
            indicator_columns.append(attr)
        else:
            raise Exception(f"Unknown attribute type: {attr}")

    df = pd.read_csv("out/train.csv")
    df = df[[a.name for a in attr_list]]

    for col_name in indicator_columns:
        categorical_column = feature_column.categorical_column_with_vocabulary_list(
            col_name.name, col_name.allowed_values)
        indicator_column = feature_column.indicator_column(categorical_column)
        feature_columns.append(indicator_column)

    train, val = train_test_split(df, test_size=0.2)

    print(feature_columns)

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)

    checkpoint_path = "census/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=0,
        save_weights_only=True,
        save_freq=1000)

    model = tf.keras.Sequential([
        feature_layer,
        # layers.Dense(64, activation='relu'),
        # layers.Dropout(0.1),
        layers.Dense(32, activation='relu'),
        # layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='RMSprop',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy', f1_m, precision_m, recall_m])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[tensorboard_callback, cp_callback],
                        verbose=1,
                        workers=8, use_multiprocessing=True)

    print(f"max val_precision: {max(history.history['val_precision_m'])}")
    print(f"average val_precision: {statistics.mean(history.history['val_precision_m'])}")
    print("")

    print(f"max val_recall: {max(history.history['val_recall_m'])}")
    print(f"average val_recall: {statistics.mean(history.history['val_recall_m'])}")
    print("")

    print(f"max val_F-value: {max(history.history['val_f1_m'])}")
    print(f"average val_F-value: {statistics.mean(history.history['val_f1_m'])}")
    print("")

    print(f"max val_accuracy: {max(history.history['val_accuracy'])}")
    print(f"average val_accuracy: {statistics.mean(history.history['val_accuracy'])}")

    df_test = pd.read_csv('out/test.csv')
    ds_test = df_to_dataset(df_test, shuffle=False, batch_size=batch_size)

    print("\n\n================== TEST ==================")

    loss, accuracy, f1_score, precision, recall = model.evaluate(ds_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}, F-Score: {f1_score}, Precision: {precision}, Recall: {recall}")
