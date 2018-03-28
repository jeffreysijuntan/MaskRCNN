# -*- coding: utf-8 -*-

import keras.preprocessing.image
import numpy
import skimage.color
import skimage.exposure
import skimage.io
import skimage.transform


class DictionaryIterator(keras.preprocessing.image.Iterator):
    def __init__(
            self,
            dictionary,
            categories,
            target_size,
            generator,
            batch_size=1,
            color_mode="rgb",
            data_format=None,
            mask_size=(28, 28),
            seed=None,
            shuffle=False
    ):
        if color_mode not in {"grayscale", "rgb"}:
            raise ValueError

        self.batch_size = batch_size

        self.categories = categories

        if color_mode == "rgb":
            self.channels = 3
        else:
            self.channels = 1

        self.color_mode = color_mode

        if data_format is None:
            data_format = keras.backend.image_data_format()

        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError

        self.data_format = data_format

        self.dictionary = dictionary

        self.generator = generator

        if self.color_mode == "grayscale":
            if self.data_format == "channels_first":
                self.image_shape = (*target_size, 1)
            else:
                self.image_shape = (1, *target_size)
        else:
            if self.data_format == "channels_last":
                self.image_shape = (*target_size, 3)
            else:
                self.image_shape = (3, *target_size)

        self.mask_size = mask_size

        self.maximum = numpy.max(target_size)

        self.minimum = numpy.min(target_size)

        self.n_categories = len(self.categories) + 1

        self.n_samples = len(self.dictionary)

        self.target_size = target_size

        super(DictionaryIterator, self).__init__(
            self.n_samples,
            batch_size,
            shuffle,
            seed
        )

    def next(self):
        with self.lock:
            selection = next(self.index_generator)

        return self._get_batches_of_transformed_samples(selection)

    def find_scale(self, image):
        r, c, _ = image.shape

        scale = self.minimum / numpy.minimum(r, c)

        if numpy.maximum(r, c) * scale > self.maximum:
            scale = self.maximum / numpy.maximum(r, c)

        return scale
    
    def _get_batches_of_transformed_samples(self, selection):
        random_crop = self.generator.random_crop
        
        horizontal_flip = False
        if self.generator.horizontal_flip:
            if numpy.random.random() < 0.5:
                horizontal_flip = True

        vertical_flip = False
        if self.generator.vertical_flip:
            if numpy.random.random() < 0.5:
                vertical_flip = True
        
        self.mask_size = (28, 28)
        
        #initialize target output sizes
        target_bounding_boxes = numpy.zeros((self.batch_size, 0, 4))

        target_categories = numpy.zeros((self.batch_size, 0, self.n_categories))

        target_images = numpy.zeros((self.batch_size, *self.target_size, self.channels))

        target_masks = numpy.zeros((self.batch_size, 0, *self.mask_size))

        target_metadata = numpy.zeros((self.batch_size, 3))
        
        for batch_idx, image_id in enumerate(selection):
            #initialize image, bounding boxes and masks
            img_fpath = self.dictionary[image_id]["image"]["pathname"]
            image = skimage.io.imread(img_fpath)[:,:,:3]
            img_r, img_c, _= image.shape
            
            if random_crop:
                dr, dc = self.target_size
                # the random min_r and min_c for the cropped image
                rand_r = numpy.random.randint(0, img_r-dr-1)
                rand_c = numpy.random.randint(0, img_c-dc-1)
                image = image[rand_r:rand_r+dr,rand_c:rand_c+dc,:]
        
            if horizontal_flip:
                image = numpy.fliplr(image)
            if vertical_flip:
                image = numpy.flipud(image)
            
            target_image = self.generator.standardize(image)
            target_images[batch_idx] = target_image
            
            target_metadata[batch_idx] = [*self.target_size, 1.0]
            
            #obtain bounding boxes from dictionary
            bounding_boxes = self.dictionary[image_id]["objects"]

            n_objects = len(bounding_boxes)
            target_bounding_boxes = numpy.resize(target_bounding_boxes, (self.batch_size, n_objects, 4))
            target_masks = numpy.resize(target_masks, (self.batch_size, n_objects, *self.mask_size))
            target_categories = numpy.resize(target_categories,(self.batch_size, n_objects, self.n_categories))

            bounding_boxes = self.dictionary[image_id]["objects"]
            for bbox_idx, bbox in enumerate(bounding_boxes):
                #transform bbox
                if bbox["class"] not in self.categories:
                    continue

                minimum_r = bbox["bounding_box"]["minimum"]["r"]
                minimum_c = bbox["bounding_box"]["minimum"]["c"]
                maximum_r = bbox["bounding_box"]["maximum"]["r"]
                maximum_c = bbox["bounding_box"]["maximum"]["c"]
                minimum_r = int(minimum_r)
                minimum_c = int(minimum_c)
                maximum_r = int(maximum_r)
                maximum_c = int(maximum_c)
                
                mask = skimage.io.imread(bbox["mask"]["pathname"])
                mask = mask[minimum_r:maximum_r, minimum_c:maximum_c]
                target_mask = skimage.transform.resize(mask, self.mask_size, order=0)

                if random_crop:
                    minimum_r = max(minimum_r-rand_r, 0)
                    maximum_r = min(maximum_r-rand_r, dr)
                    minimum_c = max(minimum_c-rand_c, 0)
                    maximum_c = min(maximum_c-rand_c, dc)
                    if maximum_r <= 0 or minimum_r >= dr or maximum_c <= 0 or minimum_r >= dc:
                        continue

                target_bounding_box = [minimum_r,minimum_c,maximum_r,maximum_c]
                if horizontal_flip:
                    target_bounding_box = [
                        target_bounding_box[0],
                        image.shape[1] - target_bounding_box[3],
                        target_bounding_box[2],
                        image.shape[1] - target_bounding_box[1]
                    ]
                    target_mask = numpy.fliplr(target_mask)
                if vertical_flip:
                    target_bounding_box = [
                        image.shape[0] - target_bounding_box[2],
                        target_bounding_box[1],
                        image.shape[0] - target_bounding_box[0],
                        target_bounding_box[3]
                    ]
                    target_mask = numpy.flipud(target_mask)
                
                target_bounding_boxes[batch_idx, bbox_idx] = target_bounding_box
                target_masks[batch_idx, bbox_idx] = target_mask
        
        target_bounding_boxes = target_bounding_boxes[numpy.newaxis,~numpy.all(target_bounding_boxes==0,axis=2)]
        target_masks = target_masks[~numpy.all(target_masks==0,axis=(2,3))]
        target_masks = numpy.expand_dims(target_masks, axis=0)
        
        x = [
            target_bounding_boxes,
            target_categories,
            target_images,
            target_masks,
            target_metadata
           ]

        return x, None
    
               
class ObjectDetectionGenerator:
    def __init__(
            self,
            data_format=None,
            horizontal_flip=False,
            preprocessing_function=None,
            rescale=False,
            rotation_range=0.0,
            samplewise_center=False,
            vertical_flip=False,
            random_crop=True
    ):
        self.data_format = data_format

        self.horizontal_flip = horizontal_flip

        self.preprocessing_function = preprocessing_function

        self.rescale = rescale

        self.rotation_range = rotation_range

        self.samplewise_center = samplewise_center

        self.vertical_flip = vertical_flip
        
        self.random_crop = random_crop

    def flow_from_dictionary(
            self,
            dictionary,
            categories,
            target_size,
            batch_size=1,
            color_mode="rgb",
            data_format=None,
            mask_size=(28, 28),
            shuffle=True,
            seed=None
    ):
        return DictionaryIterator(
            dictionary,
            categories,
            target_size,
            self,
            batch_size,
            color_mode,
            data_format,
            mask_size,
            seed,
            shuffle
        )

    def standardize(self, image):
        if self.preprocessing_function:
            image = self.preprocessing_function(image)

        if self.rescale:
            image *= self.rescale

        if self.samplewise_center:
            image = image.astype('float64')
            image -= numpy.mean(image,keepdims=True)
        
        #image = image.astype('float64')
        image = skimage.exposure.rescale_intensity(image, out_range=(0.0, 1.0))

        return image
