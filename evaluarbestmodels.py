# This script evaluates a pre-trained model on a dataset, displaying its performance metrics and confusion matrix.
import tensorflow as tf

# This configures the GPU settings to limit memory usage. In order to avoid memory issues, we set a limit of 5000MB for the first GPU detected.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)]
        )
        print("We limit the memory of the GPU to 5000MB")
    except RuntimeError as e:
        print("Error al configurar la GPU:", e)
else:
    print("No se detectó ninguna GPU.")


# Import necessary libraries
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay,  confusion_matrix
import os
from tensorflow.keras                       import regularizers
from tensorflow.keras.models                import Sequential, Model
from tensorflow.keras.layers                import *
from tensorflow.keras.callbacks             import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers            import SGD
import matplotlib.pyplot as plt

# To set the path to the dataset and the image size
DATASET_BASE_PATH = './Datasets_LSE'
# DATASET_PATH = DATASET_BASE_PATH + '/Letters192_Keypoints'
DATASET_PATH = DATASET_BASE_PATH + '/Letters512_Image_and_Keypoints'
# image_side = 192
image_side = 512
batch_size = 32

# Normalization layer
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Function to load the dataset
def load_dataset(subfolder):
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH + f"/{subfolder}",
        label_mode = "categorical",
        shuffle = False,  # No shuffle to maintain order for confusion matrix
        image_size = (image_side, image_side),
        batch_size = batch_size
    )
    return raw_ds, raw_ds.map(lambda x, y: (normalization_layer(x), y))

# Load the datasets
raw_train_ds, train_ds = load_dataset("Train") # We can change this for TrainV 
raw_val_ds, val_ds     = load_dataset("Test") # We can change this for Validation 
# raw_test_ds, test_ds   = load_dataset("Test")

class_names = raw_train_ds.class_names
print("Clases:", class_names)

# Function to get the confusion matrix and plot it
def getConfusionMatrix(model, dataset):
    y_pred = []
    y_true = []

    for image_batch, label_batch in dataset:
        y_true.append(label_batch)
        preds = model.predict(image_batch)
        y_pred.append(np.argmax(preds, axis=-1))

    correct_labels = tf.concat([item for item in y_true], axis=0)
    predicted_labels = tf.concat([item for item in y_pred], axis=0)

    final_correct_labels = []
    for lb in correct_labels:
        itlist = list(lb.numpy())
        final_correct_labels.append(itlist.index(1.0))

    # Convert to numpy arrays for confusion matrix
    cm = confusion_matrix(final_correct_labels, predicted_labels)

    # Plotting the confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        ax=ax,
        cmap=plt.cm.Blues,
        xticks_rotation=45,
        colorbar=True,
        values_format='d' 
    )
    ax.set_title("Confusion Matrix - Test", fontsize=16)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    return fig

# Load the pre-trained model
model = tf.keras.models.load_model("./best_model_final.keras")
model.summary()

# Evaluate the model on the training and validation datasets
for name, dataset in [("Train", train_ds), ("Validation", val_ds)]:
    loss, acc = model.evaluate(dataset, verbose=0)
    print(f"🔍 {name} - Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

conf_fig = getConfusionMatrix(model, val_ds)
conf_fig.savefig("confusion_matrix_validation.png")
print("✅ Confusion matrix saved as 'confusion_matrix_validation.png'")


