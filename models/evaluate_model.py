import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

def load_data(dataset_type):
    X = np.load(f'data/processed/X_{dataset_type}.npy')
    y = np.load(f'data/processed/y_{dataset_type}.npy')
    return X, y

if __name__ == '__main__':
    X_test, y_test = load_data('test')
    model = load_model('models/fer2013_model.h5')
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred_classes))
