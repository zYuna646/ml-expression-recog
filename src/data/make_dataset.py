# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    train_dir = input_filepath + '/train'
    test_dir = input_filepath + '/test'

    X_train, y_train = process_images(train_dir)
    X_test, y_test = process_images(test_dir)

    save_data(X_train, y_train, 'train', output_filepath)
    save_data(X_test, y_test, 'test', output_filepath)
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

def process_images(data_dir, target_size=(48, 48)):
    labels = []
    images = []
    for label in os.listdir(data_dir):
        if not label.startswith('.'):
            for file in os.listdir(os.path.join(data_dir, label)):
                if not file.startswith('.'):
                    img_path = os.path.join(data_dir, label, file)
                    image = load_img(img_path, target_size=target_size, color_mode='grayscale')
                    image = img_to_array(image) / 255.0
                    images.append(image)
                    labels.append(label)
    return np.array(images), pd.get_dummies(labels).values

def save_data(X, y, dataset_type, output):
    np.save(os.path.join(output, f'X_{dataset_type}.npy'), X)
    np.save(os.path.join(output, f'y_{dataset_type}.npy'), y)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
