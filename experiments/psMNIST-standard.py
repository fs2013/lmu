# Uses the exact data set and permutation from the psMNIST task in
# Towards Non-saturating Recurrent Units for Modelling Long-term Dependencies
# https://github.com/apsarath/NRU/

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'

WORK_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
nru_path = os.path.join(WORK_PATH, 'nru_project')

sys.path.append(nru_path)

import numpy as np

# from NRU.nru_project.utils.utils import create_config
# from NRU.nru_project.train import get_data_iterator
from utils.utils import create_config
from train import get_data_iterator


os.chdir(nru_path)
os.environ["PROJ_SAVEDIR"] = "/tmp/"  # this shouldn't do anything
config = create_config(os.path.join(nru_path, "config/nru.yaml"))

padded_length = 785
batch_size = 100

mask_check = np.zeros((padded_length, batch_size))
mask_check[-1, :] = 1

from collections import defaultdict
X = defaultdict(list)
Y = defaultdict(list)

gen = get_data_iterator(config)  # uses a fixed data seed
for tag in ("train", "valid", "test"):
    while True:
        data = gen.next(tag)
        if data is None:
            break

        assert data['x'].shape == (padded_length, batch_size, 1)
        assert data['y'].shape == data['mask'].shape == (padded_length, batch_size)
        assert np.all(data['mask'] == mask_check)

        assert np.all(data['x'][-1, :, :] == 0)
        X[tag].extend(data['x'][:-1, :, :].transpose(1, 0, 2))

        assert np.all(data['y'][:-1, :] == 0)
        Y[tag].extend(data['y'][-1, :])

X_train = np.asarray(X["train"])
X_valid = np.asarray(X["valid"])
X_test = np.asarray(X["test"])

Y_train = np.asarray(Y["train"])
Y_valid = np.asarray(Y["valid"])
Y_test = np.asarray(Y["test"])

print(X_train.shape, Y_train.shape)
print(X_valid.shape, Y_valid.shape)
print(X_test.shape, Y_test.shape)

print('DONE!')


from lmu import LMUCell

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense
from keras.layers.recurrent import RNN
from keras.models import Sequential, Model
from keras.initializers import Constant
from keras.utils import multi_gpu_model, to_categorical

seed = 0  # to help with reproducibility

from tensorflow import set_random_seed
set_random_seed(seed=seed)
np.random.seed(seed=seed)

n_pixels = padded_length - 1
assert n_pixels == 28**2

def lmu_layer(**kwargs):
    return RNN(LMUCell(units=212,
                       order=256,
                       theta=n_pixels,
                       input_encoders_initializer=Constant(1),
                       hidden_encoders_initializer=Constant(0),
                       memory_encoders_initializer=Constant(0),
                       input_kernel_initializer=Constant(0),
                       hidden_kernel_initializer=Constant(0),
                       memory_kernel_initializer='glorot_normal',
                      ),
               return_sequences=False,
               **kwargs)

model = Sequential()
model.add(lmu_layer(
    input_shape=X_train.shape[1:],  # (nr. of pixels, 1)
))
model.add(Dense(10, activation='softmax'))

# model = multi_gpu_model(model, gpus=4)

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.summary()

import time

epochs = 10
fname = os.path.join(WORK_PATH, 'models', 'psMNIST-standard.hdf5')

callbacks = [
    # CSVLogger('log-ctn19-pmnist-dn-090419.csv', append=True, separator=';')
    # EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=5),
    ModelCheckpoint(filepath=fname, monitor='val_loss', verbose=1, save_best_only=True),
]

t = time.time()

result = model.fit(
    X_train,
    to_categorical(Y_train),
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_valid, to_categorical(Y_valid)),
    callbacks=callbacks,
)

print("Took {:.2f} min".format((time.time() - t) / 60))

