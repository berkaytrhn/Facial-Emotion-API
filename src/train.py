from math import e
import numpy as np
import os
import json
import argparse


import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(labels):
    """
    takes labels obtained from datalooader as only parameter
    computes class weights using 'compute_class_weight' function from sklearn
    returns weight dictionary of classes to be used in model training process
    """
    labels = np.array(labels)

    weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(labels), 
        y=labels
    )
    return {index:weight for index, weight in enumerate(weights)}



def main(args):
    model_name = "5class_emotion_model"
    training_path = os.path.join(args.dataset_directory, "")
    _epochs=5
    epochs=50
    batch_size = 64
    img_height = 96
    img_width = 96

    # number of layers to open for fine tuning
    last_n_layer = 200 
    _classes = ['Happy', 'Neutral', 'Sad', 'Suprise', 'Fear']
    num_classes = len(_classes)

    train_gen = ImageDataGenerator(
        rotation_range=10,
        #width_shift_range=0.2,
        brightness_range=(0.6, 1.2),
        shear_range=0.05,
        #height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_gen = ImageDataGenerator()


    train_generator = train_gen.flow_from_directory(
        training_path+"training",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode="rgb",
        seed=1337,
        class_mode="categorical",
        classes=_classes
    )

    test_generator = test_gen.flow_from_directory(
        training_path+"validation",
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode="rgb",
        seed=1337,
        class_mode="categorical",
        classes=_classes
    )

    
    # get class weights
    print(train_generator.class_indices)
    labels = train_generator.labels
    weights = get_class_weights(labels)
    print(weights)

    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(img_height,img_width,3))


    base_model.trainable=False
    print("number of layers:",len(base_model.layers))


    def _model(base, num_classes):
        x = GlobalAveragePooling2D()(base.output)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = Dropout(0.4)(x)
        outputs = Dense(num_classes, activation="softmax")(x)
        return Model(base.input, outputs)

    model = None
    if args.load:
        model = load_model(args.load)
        model_name = args.load.split(".")[0]
    else:
        model = _model(base_model, num_classes)

    if not args.load:
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VALID = test_generator.n // test_generator.batch_size

        checkpoint = ModelCheckpoint(filepath=f"{model_name}.h5", monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
        earlystopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
        csv_logger = CSVLogger(f"{model_name}_history.log", separator=" ")

        callbacks = [checkpoint, earlystopping, rlrop, csv_logger]

        model.compile(
            optimizer=RMSprop(
                learning_rate=1e-4,
                rho=0.9,
                epsilon=1e-07,
                centered=False
            ),
            loss=CategoricalCrossentropy(),
            metrics=[
                "accuracy"
            ]
        )


        history = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=test_generator,
            validation_steps=STEP_SIZE_VALID,
            callbacks=callbacks,
            epochs=_epochs,
            class_weight=weights
        )

    for layer in (model.layers)[-last_n_layer:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable=True
            

    model.compile(
        optimizer=RMSprop(
            learning_rate=1e-5,
            rho=0.9,
            epsilon=1e-07,
            centered=False
        ),
        loss=CategoricalCrossentropy(),
        metrics=[
                "accuracy"
            ]
        )

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = test_generator.n // test_generator.batch_size

    checkpoint = ModelCheckpoint(filepath=f"{model_name}.h5", monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")
    earlystopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=1)
    csv_logger = CSVLogger(f"{model_name}_history.log", separator=" ")

    callbacks = [checkpoint, earlystopping, rlrop, csv_logger]

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=test_generator,
        validation_steps=STEP_SIZE_VALID,
        callbacks=callbacks,
        epochs=epochs,
        class_weight=weights
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--load", default=None) # value -> full name of model to be loaded
    parser.add_argument("--dataset_directory", required=True)
    args = parser.parse_args()
    
    main(args)