import cv2
import os
import json
import math

from numpy import save


def get_images_and_bounding_boxes(directories):

    images_and_boundingboxes = {}

    for dir in directories:
        files = os.listdir(dir)
        f = [dir + '/' + file for file in files if file.endswith('.json')]

        for json_file in f:
            with open(json_file, 'r') as file:
                data = json.load(file)

                images_and_boundingboxes[json_file] = data['bounding_boxes'][0]['box']

    for json_file in images_and_boundingboxes.keys():
        box = images_and_boundingboxes[json_file]
        x = box[0]
        y = box[1]
        x_offset = box[2]
        y_offset = box[3]

        center = round(x+(x_offset/2)), round(y+(y_offset/2))
        images_and_boundingboxes[json_file] = {'box': box, 'center': center}

    return images_and_boundingboxes


def save_additional_images(path, images_and_boundingboxes, label):

    for image in images_and_boundingboxes:
        image_path = image[:-4]+'jpg'
        img = cv2.imread(image_path)
        if img is not None:
            x_coord, y_coord = images_and_boundingboxes[image]['center']

            h, w = img.shape[:2]
            min_dist = min(x_coord, y_coord, w-x_coord, h-y_coord)

            crop_img = img[y_coord - min_dist:y_coord +
                           min_dist, x_coord-min_dist:x_coord+min_dist]
            dim = (256, 256)
            resized_img = cv2.resize(crop_img, dim)

            filename = os.path.split(image_path)[1]
            if(not os.path.exists(path)):
                os.mkdir(path)

            if(not os.path.exists(os.path.join(path, label))):
                os.mkdir(os.path.join(path, label))

            savePath = os.path.join(path, label, filename)
            cv2.imwrite(savePath, resized_img)


def loadAdditionalData(path, directories_and_labels):

    if(not os.path.exists(os.path.join(path))):
        os.mkdir(os.path.join(path))

    for directory in ["background", "pond", "pool", "solar", "trampoline"]:
        if(not os.path.exists(os.path.join(path, directory))):
            os.mkdir(os.path.join(path, directory))

    for directory, label in directories_and_labels:

        print("Loading data for (", directory, ", ", label, ")...")
        images_and_bounding_boxes = get_images_and_bounding_boxes([directory])
        save_additional_images(path, images_and_bounding_boxes, label)
