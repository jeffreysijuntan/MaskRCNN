# -*- coding: utf-8 -*-

import json
import os.path

import keras.utils.data_utils


def load_data(name):
    origin = "http://storage.googleapis.com/tsjbucket/{}.tar.gz".format(name)

    pathname = keras.utils.data_utils.get_file(
        fname=name,
        origin=origin,
        untar=True
    )

    training_images_pathname = os.path.join(pathname, "train")
    testing_images_pathname = os.path.join(pathname, 'test')

    masks_pathname = os.path.join(pathname, "masks")

    if not os.path.exists(masks_pathname):
        masks_pathname = None

    training_pathname = '../data/training.json'

    training = get_file_data(training_pathname, training_images_pathname, masks_pathname)

    test_pathname = '../data/testing.json'

    test = get_file_data(test_pathname, tesing_images_pathname)

    return training, test

def get_file_data(json_pathname, images_pathname, masks_pathname=None):
    if os.path.exists(json_pathname):
        with open(json_pathname) as data:
            dictionaries = json.load(data)
    else:
        dictionaries = []

    for dictionary in dictionaries:
        dictionary["image"]["pathname"] = os.path.join(images_pathname, dictionary["image"]["pathname"])

        if masks_pathname:
            for index, instance in enumerate(dictionary["objects"]):
                dictionary["objects"][index]["mask"]["pathname"] = os.path.join(masks_pathname, dictionary["objects"][index]["mask"]["pathname"])

    return dictionaries
