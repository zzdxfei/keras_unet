#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from unet_def import *
import cv2
import keras
from keras.optimizers import *


def transform_predict_to_image(predict, save_name):
    predict = predict[:, :, 0]
    result = np.zeros_like(predict, dtype=np.uint8)
    result[np.where(predict > 0.5)] = 255
    cv2.imwrite(save_name, result)


def get_test_images(image_dir, target_size=(256, 256)):
    images = []
    image_list = [os.path.join(image_dir, item) for item in os.listdir(image_dir)]
    for img_path in image_list:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, target_size)
        image = image.astype(np.float32)
        image /= 256.0
        image = image[np.newaxis, :, :, np.newaxis]
        images.append(image)
    images = np.concatenate(images)
    return images, image_list


def get_train_images_masks(image_dir, mask_dir, target_size=(256, 256)):
    images = []
    masks = []
    image_list = [os.path.join(image_dir, item) for item in os.listdir(image_dir)]
    mask_list = [os.path.join(mask_dir, item) for item in os.listdir(mask_dir)]
    assert(len(image_list) == len(mask_list))
    for img_path, mask_path in zip(image_list, mask_list):
        assert(img_path.split('/')[-1][:-3] == mask_path.split('/')[-1][:-3])
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size)

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        image /= 256.0
        mask /= 256.0
        image = image[np.newaxis, :, :, np.newaxis]
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        mask = mask[np.newaxis, :, :, np.newaxis]
        images.append(image)
        masks.append(mask)

    images = np.concatenate(images)
    masks = np.concatenate(masks)

    return (images, masks)


def make_train_generator(images, masks, batch_size=2):
    seed = 1
    data_augment_args = dict(rotation_range=60, width_shift_range=0.05,
                             height_shift_range=0.05, shear_range=0.05,
                             zoom_range=0.05, vertical_flip=True,
                             horizontal_flip=True, fill_mode='nearest')

    image_generator = keras.preprocessing.image.ImageDataGenerator(**data_augment_args)
    mask_generator = keras.preprocessing.image.ImageDataGenerator(**data_augment_args)
    image_generator = image_generator.flow(images, batch_size=batch_size, seed=seed)
    mask_generator = mask_generator.flow(masks, batch_size=batch_size, seed=seed)
    return zip(image_generator, mask_generator)


def get_callbacks():
    mcp = keras.callbacks.ModelCheckpoint(filepath='./weights/{val_loss:.5f}.hdf5',
                                          monitor='val_loss', save_best_only=False,period=10)
    tb = keras.callbacks.TensorBoard(log_dir='./logs', batch_size=2, write_grads=True, write_images=True)
    return [mcp, tb]


if __name__ == "__main__":

    train_images, train_masks = get_train_images_masks('data/membrane/train/image', 'data/membrane/train/label')
    test_images, test_image_names = get_test_images('data/membrane/test')
    valid_images, valid_masks = train_images[24:], train_masks[24:]
    train_images, train_masks = train_images[:24], train_masks[:24]

    print "train images shape : ", train_images.shape
    print "train masks shape : ", train_masks.shape
    print "valid images shape : ", valid_images.shape
    print "valid masks shape : ", valid_masks.shape
    print "test image shape : ", test_images.shape

    train_generator = make_train_generator(train_images, train_masks)

    model = unet()
    model.summary()
    # model.load_weights('./weights/best_model.hdf5')
    model.compile(optimizer=adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit_generator(train_generator, steps_per_epoch=30, epochs=50,
            validation_data = (valid_images, valid_masks), callbacks=get_callbacks())

    # test
    predict_result = model.predict(test_images, batch_size=len(test_images) / 5)
    for index, item in enumerate(predict_result):
        save_name = 'results/' + test_image_names[index].split('/')[-1]
        transform_predict_to_image(item, save_name)
