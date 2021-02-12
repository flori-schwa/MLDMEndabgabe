import datetime

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers

import arff
import preprocess

import Florian
import Frank
import Phillipp

import statistics

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


if __name__ == "__main__":
    census_raw, attributes = arff.parse_arff_file('adult_train.arff')

    tc = preprocess.TransformContext()

    if not tc.load():
        tc.calc(census_raw)
        tc.save()

    print("Transforming data...")

    tc.initialize_transformation(census_raw, attributes)

    tc.drop_attributes('fnlwgt', 'race', 'capital-loss', 'education', 'relationship')

    tc.transform_attribute(preprocess.class_to_target)

    tc.transform_attribute(Florian.transform_cntry_regional)
    tc.transform_attribute(Florian.transform_workclass)
    tc.transform_attribute(Florian.transform_hrs_per_week)
    tc.transform_attribute(Florian.transform_age)
    # tc.transform_attribute(Florian.transform_education_hs_col_grad)
    # tc.transform_attribute(Florian.transform_education_is_grad_college)

    # tc.transform_attribute(Frank.transform_age_c)
    tc.transform_attribute(Frank.transform_capital_gain_bin)

    tc.transform_attribute(Phillipp.transform_marital_status)

    attr_list = tc.transform_data('census_transformed', 'out/data_transformed.arff', 'out/data_transformed.csv')

    print("Running NN")

    feature_columns = []
    indicator_columns = []

    for attr in attr_list:
        if attr.name == "target":
            continue

        if isinstance(attr, arff.IntegerAttribute) or isinstance(attr, arff.NumericAttribute):
            feature_columns.append(feature_column.numeric_column(attr.name))
        else:
            indicator_columns.append(attr.name)

    df = pd.read_csv("out/data_transformed.csv")

    for col_name in indicator_columns:
        categorical_column = feature_column.categorical_column_with_vocabulary_list(
            col_name, df[col_name].unique())
        indicator_column = feature_column.indicator_column(categorical_column)
        feature_columns.append(indicator_column)

    train, val = train_test_split(df, test_size=0.2)

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
        layers.Dense(32, activation='sigmoid'),
        layers.Dense(16, activation='sigmoid'),
        layers.Dense(8, activation='sigmoid'),
        layers.Dense(1)
    ])

    model.compile(optimizer='RMSprop',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[tensorboard_callback], verbose=1, workers=8, use_multiprocessing=True)

    print(f"max val_accuracy: {max(history.history['val_accuracy'])}")
    print(f"average val_accuracy: {statistics.mean(history.history['val_accuracy'])}")


