# -*- coding: utf-8 -*-

import keras.backend
import keras.engine
import keras.layers

import keras_rcnn.backend
import keras_rcnn.classifiers
import keras_rcnn.layers
import keras_rcnn.preprocessing


class MaskRCNN(keras.models.Model):
    def __init__(self, input_shape, categories):
        n_categories = len(categories) + 1

        target_r, target_c, _ = input_shape

        target_bounding_boxes = keras.layers.Input(
            shape=(None, 4),
            name="target_bounding_boxes"
        )

        target_categories = keras.layers.Input(
            shape=(None, n_categories),
            name="target_categories"
        )

        target_image = keras.layers.Input(
            shape=input_shape,
            name="target_image"
        )

        target_masks = keras.layers.Input(
            shape=(None, target_r, target_c),
            name="target_images"
        )

        target_metadata = keras.layers.Input(
            shape=(3,),
            name="target_metadata"
        )

        options = {
            "activation": "relu",
            "kernel_size": (3, 3),
            "padding": "same"
        }

        inputs = [
            target_bounding_boxes,
            target_categories,
            target_image,
            target_masks,
            target_metadata
        ]

        output_features = keras.layers.Conv2D(64, **options)(target_image)

        convolution_3x3 = keras.layers.Conv2D(64, **options)(output_features)

        output_deltas = keras.layers.Conv2D(9 * 4, (1, 1), activation="linear", kernel_initializer="zero", name="deltas")(convolution_3x3)

        output_scores = keras.layers.Conv2D(9 * 1, (1, 1), activation="sigmoid", kernel_initializer="uniform", name="scores")(convolution_3x3)

        target_anchors, target_proposal_bounding_boxes, target_proposal_categories = keras_rcnn.layers.AnchorTarget()([target_bounding_boxes, target_metadata, output_scores])

        output_deltas, output_scores = keras_rcnn.layers.RPN()([target_proposal_bounding_boxes, target_proposal_categories, output_deltas, output_scores])

        output_proposal_bounding_boxes = keras_rcnn.layers.ObjectProposal()([target_anchors, target_metadata, output_deltas, output_scores])

        target_proposal_bounding_boxes, target_proposal_categories, output_proposal_bounding_boxes = keras_rcnn.layers.ProposalTarget()([target_bounding_boxes, target_categories, output_proposal_bounding_boxes])

        output_features = keras_rcnn.layers.RegionOfInterest((14, 14))([target_metadata, output_features, output_proposal_bounding_boxes])

        output_features = keras.layers.TimeDistributed(keras.layers.Flatten())(output_features)

        output_features = keras.layers.TimeDistributed(keras.layers.Dense(256, activation="relu"))(output_features)

        # Bounding Boxes
        output_deltas = keras.layers.TimeDistributed(keras.layers.Dense(4 * n_categories, kernel_initializer="zero"))(output_features)

        output_deltas = keras.layers.Activation("linear")(output_deltas)

        # Categories
        output_scores = keras.layers.TimeDistributed(keras.layers.Dense(1 * n_categories, kernel_initializer="zero"))(output_features)

        output_scores = keras.layers.Activation("softmax")(output_scores)

        output_deltas, output_scores = keras_rcnn.layers.RCNN()([target_proposal_bounding_boxes, target_proposal_categories, output_deltas, output_scores])

		#Mask branch
		output_masks = keras.layers.TimeDistributed(
    		keras.layers.Conv2D(filters=256, **options)
		)(output_features)

        output_masks = keras.layers.TimeDistributed(
            keras.layers.Conv2D(filters=256, **options)
        )(output_masks)

        output_masks = keras.layers.TimeDistributed(
            keras.layers.Conv2D(filters=256, **options)
        )(output_masks)

		output_masks = keras.layers.TimeDistributed(
    		keras.layers.Conv2D(
        		activation="sigmoid",
        		filters=n_categories, 
        		kernel_size=(1, 1), 
        		strides=1
    		)
		)(output_masks)

		output_masks = keras_rcnn.layers.MaskRCNN()([target_proposal_categories, target_masks, output_masks])

        output_bounding_boxes, output_categories = keras_rcnn.layers.ObjectDetection()([target_metadata, output_deltas, output_proposal_bounding_boxes, output_scores])

        outputs = [
            output_bounding_boxes,
            output_categories,
            output_masks
        ]

        super(MaskRCNN, self).__init__(inputs, outputs)

    def compile(self, optimizer, **kwargs):
        super(MaskRCNN, self).compile(optimizer, None)
