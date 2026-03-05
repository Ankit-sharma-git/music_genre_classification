import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.cnn_model import build_cnn

def train_cnn(data_path="data/raw/Data/images_original"):
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2
    )
    
    train_generator = datagen.flow_from_directory(
        data_path,
        target_size=(128,128),
        batch_size=32,
        class_mode="categorical",
        subset="training"
    )
    
    val_generator = datagen.flow_from_directory(
        data_path,
        target_size=(128,128),
        batch_size=32,
        class_mode="categorical",
        subset="validation"
    )
    
    model = build_cnn((128, 128, 3), train_generator.num_classes)
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20
    )
    
    model.save("models/saved_models/cnn_model.h5")

