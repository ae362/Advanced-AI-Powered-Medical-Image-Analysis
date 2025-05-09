from django.core.management.base import BaseCommand
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from django.conf import settings
import multiprocessing

class Command(BaseCommand):
    help = 'Improve the cancer detection model'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting improved cancer model training...'))

        train_ds, val_ds, class_names = self.create_cancer_datasets('data/cancer')

        num_classes = len(class_names)

        self.stdout.write(f"Training cancer model with {num_classes} classes: {class_names}")

        model, history = self.train_cancer_model(train_ds, val_ds, num_classes)

        self.stdout.write(self.style.SUCCESS("\nEvaluating improved cancer model..."))
        report = self.evaluate_cancer_model(model, val_ds, class_names)

        self.plot_cancer_training_history(history)

        self.stdout.write(self.style.SUCCESS("\nFinal Cancer Model Performance Metrics:"))
        self.stdout.write(str(pd.DataFrame(report).T))

    def create_advanced_cancer_model(self, input_shape=(224, 224, 3), num_classes=3):
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape
        )
        
        for layer in base_model.layers[-30:]:
            layer.trainable = True
        
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        return tf.keras.Model(inputs, outputs)

    def create_cancer_datasets(self, directory, batch_size=32, validation_split=0.2):
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])
        
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=True
        )
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=True
        )
        
        class_names = train_ds.class_names
        
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), tf.one_hot(y, depth=len(class_names))),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        val_ds = val_ds.map(
            lambda x, y: (x, tf.one_hot(y, depth=len(class_names))),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds, class_names

    def train_cancer_model(self, train_ds, val_ds, num_classes, epochs=50):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = self.create_advanced_cancer_model(num_classes=num_classes)
            
            initial_learning_rate = 1e-4
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=10000,
                decay_rate=0.9,
                staircase=True
            )
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                str(Path(settings.BASE_DIR) / 'api' / 'ml' / 'saved_models' / 'cancer_improved_model.h5'),
                save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
        ]
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            workers=multiprocessing.cpu_count(),
            use_multiprocessing=True
        )
        
        return model, history

    def evaluate_cancer_model(self, model, dataset, class_names):
        y_pred = []
        y_true = []
        
        for images, labels in dataset:
            predictions = model.predict(images)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(np.argmax(labels.numpy(), axis=1))
        
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Cancer Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(str(Path(settings.BASE_DIR) / 'api' / 'ml' / 'analysis' / 'cancer_improved_confusion_matrix.png'))
        plt.close()
        
        return report

    def plot_cancer_training_history(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Cancer Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Cancer Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(str(Path(settings.BASE_DIR) / 'api' / 'ml' / 'analysis' / 'cancer_improved_training_history.png'))
        plt.close()

