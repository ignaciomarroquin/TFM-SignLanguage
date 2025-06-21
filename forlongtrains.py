import tensorflow as tf

# This code is to limit the GPU memory usage to 14000MB (14GB) if a GPU is available. (It can be adjusted as needed.)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14000)]
        )
        print("We limit the memory of the GPU to 5000MB")
    except RuntimeError as e:
        print("Error with the GPU:", e)
else:
    print("No GPU detected.")

# Import necessary libraries
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import os
from tensorflow.keras                       import regularizers
from tensorflow.keras.models                import Sequential, Model
from tensorflow.keras.layers                import *
from tensorflow.keras.callbacks             import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers            import SGD
import matplotlib.pyplot as plt


def getConfusionMatrix(model, dataset):
    y_pred = []  
    y_true = []  

    for image_batch, label_batch in dataset:   
        
        y_true.append(label_batch) 
        preds = model.predict(image_batch) 
        y_pred.append(np.argmax(preds, axis = - 1)) 
 
    correct_labels = tf.concat([item for item in y_true], axis = 0)
    predicted_labels = tf.concat([item for item in y_pred], axis = 0)

    final_correct_labels = []
    for lb in correct_labels:
        itlist=list(lb.numpy())
        final_correct_labels.append(itlist.index(1.0))

    return ConfusionMatrixDisplay.from_predictions(final_correct_labels, predicted_labels, display_labels=class_names, cmap=plt.cm.Blues)

def logModelTrainFinished(run, model, validation_ds, class_names): 

    os.makedirs("model_trainer", exist_ok=True) 

    model.save("model_trainer/current_run_model.keras") 
    run["model_last"].upload("model_trainer/current_run_model.keras")

    getConfusionMatrix(model, validation_ds).figure_.savefig("model_trainer/confusion_matrix.png") 
    run["eval/conf_matrix"].upload("model_trainer/confusion_matrix.png") 

    
    if os.path.exists("model_trainer/missclasified"):
        os.system("rm -rf model_trainer/missclasified")
    os.makedirs("model_trainer/missclasified", exist_ok=True)

    cubatch = 0 
    for image_batch, label_batch in validation_ds: 
    
        preds = model.predict(image_batch)

        for i in range(len(preds)):
            if np.argmax(preds[i]) != np.argmax(label_batch[i]):
                imgName = "batch_" + str(cubatch) + "_img_" + str(i) +  "_true_" + class_names[np.argmax(label_batch[i])] + "_pred_" + class_names[np.argmax(preds[i])] + ".png"
                imgPath = "missclasified/" + imgName
                plt.imsave("model_trainer/" + imgPath, image_batch[i].numpy())
                run["eval/"+imgPath].upload("model_trainer/" + imgPath)
    
        cubatch += 1

    try:
        accuracy_dataframe = run["eval/accuracy"].fetch_values() 
    except:
        accuracy_dataframe = []
    
    run["train/epochs_executed"] = len(accuracy_dataframe) 
    run["model_params_count"] = model.count_params() 

    model = tf.keras.models.load_model("model_best.keras")

    model.save("model_trainer/best_run_model.keras") 
    run["best_model"].upload("model_trainer/best_run_model.keras")

    first = accuracy_dataframe.iloc[0] 
    last = accuracy_dataframe.iloc[-1]

    first = first["timestamp"]
    last = last["timestamp"]

    duration_timedelta = last - first

    run["train/duration_seconds"] = duration_timedelta.total_seconds()
    run["train/duration_text"] = str(duration_timedelta)

    run.wait()
    run.stop()

DATASET_BASE_PATH = './Datasets_LSE'
DATASET_PATH = DATASET_BASE_PATH + '/Letters512_Keypoints'

neptune_api_token = open("./neptune_api_token_LSE.txt").read()
using_neptune = True

import neptune
os.environ["NEPTUNE_API_TOKEN"] = neptune_api_token

bs = 12 # It's the batch size. How many images are loaded per batch during training. # Lo he cambiado de 64 a 32 para ver si así funcionaba

image_side = 512 # (THIS IS IMPORTANT, In Letters192 there are some images with a image size greater than 192x192 but by doing this we fix it)

with tf.device('/cpu:0'):
  raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH + "/Train", # /TrainV IF WE WANT TO USE THE TRAIN/VALIDATION SPLIT
    label_mode = "categorical",
    shuffle = True,
    image_size = (image_side, image_side),
    batch_size = bs)

  raw_validation_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH + "/Test", # /Validation IF WE WANT TO USE THE TRAIN/VALIDATION SPLIT
    label_mode = "categorical",
    shuffle = True,
    image_size = (image_side, image_side),
    batch_size = bs)
  

class_names = raw_train_ds.class_names
print(class_names)

# Normalize from 0-255 to 0-1
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = raw_train_ds.map(lambda x, y: (normalization_layer(x), y))
validation_ds = raw_validation_ds.map(lambda x, y: (normalization_layer(x), y))

# It creates realistic variations of your training images (on the fly) so the model doesn’t overfit to the training data, learns to be robust to noise, rotations, translations, etc.

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.01),
        tf.keras.layers.RandomZoom(0.02),
        tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode='nearest', fill_value=0.5),
        tf.keras.layers.RandomBrightness([-0.15,0.1],value_range=(0, 1)),
        #tf.keras.layers.RandomCrop(25,25),
        #tf.keras.layers.RandomContrast(0.02)
    ]
)
data_augmentation.build((None, image_side, image_side, 3)) # This is important to prevent usage of data augmentation change the shape

# Here we config model params: (That I selected from the Hyperparameter Tuning of Pamela)
model_params = { "dropout1": 0.3, "dropout2": 0.2, "dense": 74, "l2reg": 0.02} # LOS DE DENSENET201
# model_params = { "dropout1": 0.3, "dropout2": 0.1, "dense": 64, "l2reg": 0.015} # LOS DE MOBILENETV2 
# model_params = { "dropout1": 0.32, "dropout2": 0.11, "dense": 66, "l2reg": 0.015} # LOS DE INCEPTIONV3 
# model_params = { "dropout1": 0.25, "dropout2": 0.1, "dense": 69, "l2reg": 0.02} # LOS DE RESNET152

project = neptune.init_project(project="LSE-Sign-Language2/LSE-Sign-Language", api_token=neptune_api_token)
os.environ["NEPTUNE_API_TOKEN"] = neptune_api_token
os.environ["NEPTUNE_PROJECT"] = "LSE-Sign-Language2/LSE-Sign-Language"

base_model = tf.keras.applications.DenseNet201(input_shape=(image_side,image_side,3), include_top=False, weights='imagenet')
#base_model = tf.keras.applications.MobileNetV2(input_shape=(image_side,image_side,3), include_top=False, weights='imagenet')
#base_model = tf.keras.applications.InceptionV3(input_shape=(image_side,image_side,3), include_top=False, weights='imagenet')
#base_model = tf.keras.applications.ResNet152(input_shape=(image_side,image_side,3), include_top=False, weights='imagenet')


num_layers_to_freeze = 140 # Number of layers to freeze, this is the number of layers that will not be trained (to avoid overfitting and speed up training)
for layer in base_model.layers[:num_layers_to_freeze]:
    layer.trainable = False


# Create the model

model = tf.keras.Sequential([
  data_augmentation, # Apply augmentation before training
  base_model,  # Feature extractor
  Flatten(),   # Flatten feature map to vector
  Dropout(model_params["dropout1"]), # Drop some neurons to avoid overfitting
  Dense(model_params["dense"], kernel_regularizer=regularizers.l2(model_params["l2reg"]), activation = 'relu'),  # Fully connected layer (with L2 regularization)
  Dropout(model_params["dropout2"]), # Drop some neurons to avoid overfitting
  Dense(len(class_names), activation = 'softmax') # Output layer: one neuron per class
])

model.build((None, image_side, image_side, 3))
model.summary()


# Here, we compile the model, initialize a Neptune run to track training, define callbacks (checkpointing, early stopping, etc.)
# train the model with model.fit(...), logs metrics to NeptuneHandles interruptions or crashes gracefully and finalizes the run with post-training logs

learning_rate = 0.003 # La de DENSENET201
# learning_rate = 0.001 # La de MOBILENETV2
# learning_rate = 0.004 # La de INCEPTIONV3
# learning_rate = 0.003 # La de RESNET152

epochs = 50 # it will try to train for 50 full passes over the dataset (unless stopped earlier).

# Starts a new Neptune run to track the training session.
if using_neptune:
    run = neptune.init_run(project="LSE-Sign-Language2/LSE-Sign-Language", api_token=neptune_api_token, capture_hardware_metrics=True, capture_stdout=True, capture_stderr=False)
    params = {
        "learning_rate": learning_rate, 
        "optimizer": "SGD",
        "base_model": base_model.name,
        "image_side": image_side,
        "epochs": epochs,
        "batch_size": bs
        }
    
    run["sys/name"] = f"{base_model.name}_freeze_{num_layers_to_freeze}_on_{os.path.basename(DATASET_PATH)}"
    run["parameters"] = params
    run["status"] = "running"
    



# categorical_crossentropy: used for multi-class classification (with one-hot labels) and SGD (Stochastic Gradient Descent) a basic optimizer.
model.compile(loss="categorical_crossentropy", optimizer= SGD(learning_rate=learning_rate), metrics=['accuracy'])

# This save the best model
checkpointer = ModelCheckpoint(filepath='model_best.keras', verbose=1, save_best_only=True, monitor = 'val_accuracy', mode = 'max')

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001) # this reduce learning rate when val_loss is not improving

# Stops training early if the validation loss doesn't improve for 5 epochs
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) 

# This function logs training and validation metrics to Neptune after every epoch
def epochCallback(epoch, logs):
    if using_neptune:
        run["train/loss"].log(logs["loss"])
        run["train/accuracy"].log(logs["accuracy"])
        run["eval/loss"].log(logs["val_loss"])
        run["eval/accuracy"].log(logs["val_accuracy"])

keyinterrupt = False
history = None

# This is the actual training loop, uses the normalized, augmented datasets. Uses callbacks to: log to Neptune, save best model, stop early if overfitting
try:
    history = model.fit(train_ds, validation_data = validation_ds, epochs=epochs,
                                callbacks = [
#                                    reduce_lr, 
                                    checkpointer,
                                    tf.keras.callbacks.LambdaCallback(on_epoch_end=epochCallback),
                                    early_stop
                                ])
    if using_neptune:
        run["status"] = "finished"
except tf.errors.ResourceExhaustedError:
    if using_neptune:
        run["status"] = "crashed-ResourceExhausted"
except KeyboardInterrupt:
    keyinterrupt=True
    if using_neptune:
        run["status"] = "Interrupted keyboard"

logModelTrainFinished(run, model, validation_ds, class_names)

if keyinterrupt:
    raise KeyboardInterrupt
