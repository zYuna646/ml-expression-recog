import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

current_dir = os.path.dirname(os.path.abspath(__file__))

def load_data(dataset_type):
    data_dir = os.path.abspath(os.path.join(current_dir, '../../data/processed'))
    X = np.load(os.path.join(data_dir, f'X_{dataset_type}.npy'))
    y = np.load(os.path.join(data_dir, f'y_{dataset_type}.npy'))
    return X, y

def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
    return history

if __name__ == '__main__':
    X_train, y_train = load_data('train')
    X_test, y_test = load_data('test')
    input_shape = (48, 48, 1)
    model = build_model(input_shape)
    history = train_model(model, X_train, y_train, X_test, y_test)
    model_dir = os.path.abspath(os.path.join(current_dir, '../../models'))
    model.save(os.path.join(model_dir, 'fer2013_model.h5'))
