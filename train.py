# Import required packages and libraries.
import numpy as np
from tensorflow.keras.layers import (Input, Dense, Dropout, Activation, AveragePooling3D, BatchNormalization, Conv3D, Flatten, concatenate, GlobalAveragePooling3D, ReLU)
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint)
from tensorflow.keras.regularizers import l2

# Cropped three-dimensional cubic patches were transformed to 50×50×50-isotropic voxel images using spline interpolation
# and normalized to a range of 0-1 using Hounsfield units of -1200 and 300 as the lower and upper bounds, respectively.
image_size = 50
MIN_BOUND = -1200.0
MAX_BOUND = 300.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

# DenseNet architecture modules
def dense_factor(inputs, kernel1):
    h_1 = BatchNormalization()(inputs)
    h_1 = Conv3D(kernel1, (3, 3, 3), kernel_initializer='he_uniform', padding='same')(h_1)
    output = ReLU()(h_1)
    return output

def transition(x, kernel2, droprate, weight_decay):
    x = BatchNormalization(axis=-1, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = ReLU()(x)
    x = Conv3D(kernel2, (1, 1, 1),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(droprate)(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
    return x

def dense_block(inputs, numlayers, kernel1):
    concatenated_inputs = inputs

    for i in range(numlayers):
        x = dense_factor(concatenated_inputs, kernel1)
        concatenated_inputs = concatenate([concatenated_inputs, x], axis=-1)

    return concatenated_inputs

def DenseNet(image_size, kernel1, kernel2, kernel3, numlayers, droprate, addD, weight_decay):
    model_input1 = Input((image_size, image_size, image_size, 1), name='image')
    x = BatchNormalization()(model_input1)
    x = Conv3D(kernel3, (3, 3, 3),
               kernel_initializer="he_uniform",
               name="initial_conv3D",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = dense_block(x, numlayers, kernel1)
    x = transition(x, kernel2, droprate, weight_decay)
    x = dense_block(x, numlayers, kernel1)
    x = transition(x, kernel2, droprate, weight_decay)
    if addD == 1:
        x = dense_block(x, 1, kernel1)
    x = BatchNormalization(axis=-1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = ReLU()(x)
    x = GlobalAveragePooling3D()(x)

    output_VPI = Dense(1, activation = 'sigmoid', kernel_initializer='zeros', bias_initializer='zeros', name = 'VPI_output')(x)
    output_subtype = Dense(1, activation='sigmoid', kernel_initializer='zeros', bias_initializer='zeros', name = 'subtype_output')(x)
    output_radtype = Dense(1, activation='sigmoid', kernel_initializer='zeros', bias_initializer='zeros', name = 'radtype_output')(x)
    output_LVI = Dense(1, activation='sigmoid', kernel_initializer='zeros', bias_initializer='zeros', name = 'LVI_output')(x)
    output_node = Dense(1, activation='sigmoid', kernel_initializer='zeros', bias_initializer='zeros',
                       name='node_output')(x)

    model = Model(inputs= model_input1, outputs=[output_VPI, output_subtype, output_radtype, output_LVI, output_node])
    return model

kernel1=32
kernel2=64
kernel3=32
addD=1

# Hyperparameters were optimized using random search
droprate=0.2
numlayers=3
weight_decay=1E-4

patience=30
batch_size=16
epochs=100

model = DenseNet(image_size, kernel1, kernel2, kernel3, numlayers, droprate, addD, weight_decay)

# Multiple output with equal loss weights
loss = {'VPI_output' : 'binary_crossentropy',
        'subtype_output' : 'binary_crossentropy',
        'radtype_output' : 'binary_crossentropy',
        'LVI_output' : 'binary_crossentropy',
        'node_output': 'binary_crossentropy'
       }

loss_weights = {'VPI_output' : 0.2,
               'subtype_output' : 0.2,
               'radtype_output' : 0.2,
               'LVI_output' : 0.2,
               'node_output': 0.2
               }

# Dynamic learning rate
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=1e-6)

optimizer = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

model_checkpoint = ModelCheckpoint(filepath=weightcache_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

history = model.fit(train_images, [train_VPI, train_Subtype, train_Radtype, train_LVI, train_Node], batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_data=(tune_images, [tune_VPI, tune_Subtype, tune_Radtype, tune_LVI, tune_Node]),
                    callbacks=[model_checkpoint, lr_reducer, early_stopping])

loaded_model = load_model(weightcache_path)

model_json = loaded_model.to_json()
with open(model_path, mode='w') as json_file:
    json_file.write(model_json)
loaded_model.save_weights(weight_path)
