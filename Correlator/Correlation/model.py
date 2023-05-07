import os
import pickle
import time

from tqdm.notebook import tqdm
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.numpy_ops import np_config

tf.config.run_functions_eagerly(True)
np_config.enable_numpy_behavior()
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(os.getenv('TF_GPU_ALLOCATOR'))

from process_flows import load_data

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001

DIR_CHECKPOINT = '/content/gdrive/MyDrive/TorDeanonymization/checkpoints/'
DIR_MODELS = '/content/gdrive/MyDrive/TorDeanonymization/models/'
NAME_CHECKPOINT = 'train_last'
HISTORY_FILE = 'history.pkl'

DEFAULT_MODEL_PARAMS = {
    'conv_filters': [2000, 1000],  # filters for the first two conv layers
    'dense_layers': [49600, 2000, 800, 100, 1],  # units for the last dense layers
    'drop_p': 0.6  # dropout rate
}


def create_model(params):
    inputs = Input(shape=(8, 300, 1), name='inputs')

    layer = Conv2D(params['conv_filters'][0], kernel_size=[2, 30], strides=(2, 1), activation='relu')(inputs)
    layer = MaxPool2D([1, 5], strides=(1, 1))(layer)
    layer = Conv2D(params['conv_filters'][1], kernel_size=[2, 10], strides=(4, 1), activation='relu')(layer)
    layer = MaxPool2D([1, 5], strides=(1, 1))(layer)
    layer = Flatten()(layer)
    # layer = Dense(params['dense_layers'][0], activation='relu')(layer)
    # layer = Dropout(params['drop_p'])(layer)
    layer = Dense(params['dense_layers'][1], activation='relu')(layer)
    layer = Dropout(params['drop_p'])(layer)
    layer = Dense(params['dense_layers'][2], activation='relu')(layer)
    layer = Dropout(params['drop_p'])(layer)
    layer = Dense(params['dense_layers'][3], activation='relu')(layer)
    layer = Dense(params['dense_layers'][4])(layer)

    return Model(inputs=inputs, outputs=layer)


def loss_function(props, y_truth):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=props, labels=y_truth))


def accuracy(logits, y_truth):
    batch_size = y_truth.shape[0]
    return np.sum((logits >= 0.5) == y_truth) / batch_size


@tf.function
def train_step(model, x, y, opt):
    # Record the operations run during the forward pass
    with tf.GradientTape() as tape:
        # Compute the loss value for this minibatch
        logits = model(x.astype('float32'), training=True)
        logits = tf.reshape(logits, [-1])  # Specific for tf.nn.sigmoid_cross_entropy_with_logits
        loss_value = loss_function(logits, y.astype('float32'))
    # Retrieve the gradients of the trainable variables
    grads = tape.gradient(loss_value, model.trainable_weights)
    # Update the value of the variables to minimize the loss
    opt.apply_gradients(zip(grads, model.trainable_weights))
    # Calculate the training metric
    acc = accuracy(tf.sigmoid(logits).numpy(), y.numpy())
    return loss_value, acc


@tf.function
def test_step(model, x, y):
    val_logits = model(x.astype('float32'), training=False)
    val_logits = tf.reshape(val_logits, [-1])
    loss = loss_function(val_logits, y.astype('float32'))
    acc = accuracy(tf.sigmoid(val_logits).numpy(), y.numpy())
    return loss, acc


def train(model, train_dataset, val_dataset):
    # Instantiate Adam optimizer to train the model
    optimizer = Adam(learning_rate=LEARNING_RATE)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=DIR_CHECKPOINT,
                                         checkpoint_name=NAME_CHECKPOINT, max_to_keep=1)

    if os.path.exists(DIR_CHECKPOINT + HISTORY_FILE):
        print("Found Checkpoint")
        checkpoint.restore(manager.latest_checkpoint)
        history = pickle.load(open(DIR_CHECKPOINT + HISTORY_FILE, 'rb'))
        print(history)
    else:
        history = {
            'epoch': 0,
            'train_acc': [],
            'train_loss': [],
            'val_acc': [],
            'val_loss': []
        }

    for epoch in tqdm(range(history['epoch'], EPOCHS)):
        print("\nStart of epoch", epoch)
        start_time = time.time()

        acc_aux = []
        loss_aux = []
        # Iterate over the batches of the dataset
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value, acc_val = train_step(model, x_batch_train, y_batch_train, optimizer)
            acc_aux.append(acc_val)
            loss_aux.append(loss_value)

            # Log every 200 batches
            if step % 200 == 0:
                print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                print("Seen so far: %d samples" % ((step + 1) * BATCH_SIZE))

        # Display the metrics at the end of each epoch
        train_acc = sum(acc_aux) / len(acc_aux)
        train_loss = sum(loss_aux) / len(loss_aux)
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)

        print("Training acc over epoch: %.4f" % (float(train_acc),))

        acc_aux = []
        loss_aux = []
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            loss_val, acc_val = test_step(model, x_batch_val, y_batch_val)
            acc_aux.append(acc_val)
            loss_aux.append(loss_val)

        val_acc = sum(acc_aux) / len(acc_aux)
        val_loss = sum(loss_aux) / len(loss_aux)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))

        manager.save()
        pickle.dump(history, open(DIR_CHECKPOINT + HISTORY_FILE, 'wb'))

        history['epoch'] += 1

    return model


def save(model, name):
    model.summary()
    model.save(DIR_MODELS + name + '.h5')
    print("Stored model.")


def predict(model, x_pred):
    props = model(x_pred.astype('float32'), training=False)
    return props


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data("pcaps")
    print("Got the training and testing data")

    test_params = {
        'conv_filters': [100, 40],  # filters for the first two conv layers
        'dense_layers': [1000, 500, 100, 10, 1],  # units for the last dense layers
        'drop_p': 0.6  # dropout rate
    }

    deepcorr = create_model(test_params)
    print("Created Model.")

    dc_train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dc_train_dataset = dc_train_dataset.batch(BATCH_SIZE)

    dc_val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dc_val_dataset = dc_val_dataset.batch(BATCH_SIZE)

    deepcorr = train(deepcorr, dc_train_dataset, dc_val_dataset)

    save(deepcorr, 'deepcorr')
