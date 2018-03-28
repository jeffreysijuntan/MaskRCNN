import numpy as np
import keras_rcnn.preprocessing._object_detection
import keras_rcnn.datasets.dsb2018
import keras_rcnn.models
import keras
import skimage.io
import sklearn.model_selection
from sklearn.model_selection import KFold

def main():
    epoch_count = 0
    for i in range(20):
        training, testing = keras_rcnn.datasets.dsb2018.load_data()
        training = np.array(training)
        np.random.shuffle(training)

        classes = {"nuclei": 1}

        kf = KFold(n_splits=5)

        for train_index, val_index in kf.split(training):
            print("Start training on epoch {}".format(epoch_count+1))

            train_data, val_data = training[train_index], training[val_index]

            train_generator = keras_rcnn.preprocessing.ObjectDetectionGenerator(horizontal_flip=True, vertical_flip=True)
            train_generator = train_generator.flow_from_dictionary(train_data, classes, target_size=(224, 224))

            val_generator = keras_rcnn.preprocessing.ObjectDetectionGenerator()
            val_generator = val_generator.flow_from_dictionary(val_data, classes, (224, 224))

            checkpointer = keras.callbacks.ModelCheckpoint(filepath='./weights/maskrcnn_weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=0, save_best_only=False)

            model = keras_rcnn.models.MaskRCNN((224,224,3), ['nuclei'])
            optimizer = keras.optimizers.Adam(0.0001)
            model.compile(optimizer)

            model.fit_generator(
                            epochs=100,
                            generator=train_generator,
                            steps_per_epoch=len(train_index),
                            callbacks=[checkpointer],
                            validation_data=val_generator,
                            validation_steps=len(val_index),
                            initial_epoch=epoch_count
                )

            epoch_count += 1

if __name__ == '__main__':
    main()