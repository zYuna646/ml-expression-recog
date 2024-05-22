import os
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))

def load_data(dataset_type):
    data_dir = os.path.abspath(os.path.join(current_dir, '../../data/processed'))
    X = np.load(os.path.join(data_dir, f'X_{dataset_type}.npy'))
    y = np.load(os.path.join(data_dir, f'y_{dataset_type}.npy'))
    return X, y

if __name__ == '__main__':
    X_test, y_test = load_data('test')
    model_dir = os.path.abspath(os.path.join(current_dir, '../../models'))

    model = load_model(os.path.join(model_dir, 'fer2013_model.h5'))
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred_classes))
