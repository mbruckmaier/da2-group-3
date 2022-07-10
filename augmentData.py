import os
from posixpath import dirname
import numpy as np
import os
from PIL import Image, ImageEnhance


def rotateDirectory(directory, folder):

    directory = directory + folder
    for file in os.scandir(directory):
        filepath = os.fsdecode(file)
        pathname, extension = os.path.splitext(filepath)

        if(filepath.endswith(".jpg") or filepath.endswith(".png")):
            img = Image.open(filepath)
            img_180 = img.rotate(180, expand=0)
            img_90 = img.rotate(90, expand=0)
            img_270 = img.rotate(270, expand=0)
            filename = os.path.split(filepath)[1].split(".")[0]

            img_180.save("training_patches_rotation/" + folder +
                         "/" + filename + "_180" + ".png")
            img_90.save("training_patches_rotation/" +
                        folder + "/" + filename + "_90" + ".png")
            img_270.save("training_patches_rotation/" + folder +
                         "/" + filename + "_270" + ".png")


def moveImageContentForDirectory(directory, folder):

    directory = directory + folder

    for file in os.scandir(directory):
        filepath = os.fsdecode(file)
        pathname, extension = os.path.splitext(filepath)

        if(filepath.endswith(".jpg") or filepath.endswith(".png")):
            img = Image.open(filepath)

            # left55
            left = 55
            top = 0
            right = 256
            bottom = 256
            im1 = img.crop((left, top, right, bottom))
            im2 = Image.new('RGB', (55, 256))
            left55img = Image.new('RGB', (im1.width + im2.width, im1.height))
            left55img.paste(im1, (0, 0))
            left55img.paste(im2, (im1.width, 0))

            # left25
            left = 25
            top = 0
            right = 256
            bottom = 256
            im1 = img.crop((left, top, right, bottom))
            im2 = Image.new('RGB', (25, 256))
            left25img = Image.new('RGB', (im1.width + im2.width, im1.height))
            left25img.paste(im1, (0, 0))
            left25img.paste(im2, (im1.width, 0))

            # right
            left = 0
            top = 0
            right = 181
            bottom = 256
            im2 = img.crop((left, top, right, bottom))
            im1 = Image.new('RGB', (75, 256))
            right75img = Image.new('RGB', (im1.width + im2.width, im1.height))
            right75img.paste(im1, (0, 0))
            right75img.paste(im2, (im1.width, 0))

            # right
            left = 0
            top = 0
            right = 201
            bottom = 256
            im2 = img.crop((left, top, right, bottom))
            im1 = Image.new('RGB', (55, 256))
            rightimg = Image.new('RGB', (im1.width + im2.width, im1.height))
            rightimg.paste(im1, (0, 0))
            rightimg.paste(im2, (im1.width, 0))

            # down
            left = 0
            top = 0
            right = 256
            bottom = 201

            im2 = img.crop((left, top, right, bottom))
            im1 = Image.new('RGB', (256, 55))
            downimg = Image.new('RGB', (im1.width, im1.height + im2.height))
            downimg.paste(im1, (0, 0))
            downimg.paste(im2, (0, im1.height))

            # up
            left = 0
            top = 55
            right = 256
            bottom = 256
            im1 = img.crop((left, top, right, bottom))
            im2 = Image.new('RGB', (256, 55))
            upimg = Image.new('RGB', (im1.width, im1.height + im2.height))
            upimg.paste(im1, (0, 0))
            upimg.paste(im2, (0, im1.height))

            # up75
            left = 0
            top = 75
            right = 256
            bottom = 256
            im1 = img.crop((left, top, right, bottom))
            im2 = Image.new('RGB', (256, 75))
            up75img = Image.new('RGB', (im1.width, im1.height + im2.height))
            up75img.paste(im1, (0, 0))
            up75img.paste(im2, (0, im1.height))

            filename = os.path.split(filepath)[1].split(".")[0]

            left55img.save("training_patches_left/" + folder +
                           "/" + filename + "_left_55" + ".png")
            left25img.save("training_patches_left/" + folder +
                           "/" + filename + "_left_25" + ".png")
            rightimg.save("training_patches_right/" + folder +
                          "/" + filename + "_right_55" + ".png")
            right75img.save("training_patches_right/" + folder +
                            "/" + filename + "_right_75" + ".png")
            downimg.save("training_patches_down/" + folder +
                         "/" + filename + "_down_55" + ".png")
            upimg.save("training_patches_up/" + folder +
                       "/" + filename + "_up_55" + ".png")
            up75img.save("training_patches_up/" + folder +
                         "/" + filename + "_up_75" + ".png")


def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


def zoomImagesForDirectory(directory, folder):

    directory = directory + folder

    for file in os.scandir(directory):
        filepath = os.fsdecode(file)
        pathname, extension = os.path.splitext(filepath)

        if(filepath.endswith(".jpg") or filepath.endswith(".png")):
            img = Image.open(filepath)

            z = zoom_at(img, 128, 128, 1.5)

            filename = os.path.split(filepath)[1].split(".")[0]

            z.save("training_patches_zoom/" + folder +
                   "/" + filename + "_zoom1_5" + ".png")


def changeBrightnessForDirectory(directory, folder):

    directory = directory + folder

    for file in os.scandir(directory):
        filepath = os.fsdecode(file)
        pathname, extension = os.path.splitext(filepath)

        if(filepath.endswith(".jpg") or filepath.endswith(".png")):
            img = Image.open(filepath)
            # image brightness enhancer
            enhancer = ImageEnhance.Brightness(img)

            factor = 0.8  # darkens the image
            im_dark = enhancer.enhance(factor)

            factor = 1.2  # brightens the image
            im_bright = enhancer.enhance(factor)

            filename = os.path.split(filepath)[1].split(".")[0]

            im_bright.save("training_patches_brightnessup/" +
                           folder + "/" + filename + "_bup" + ".png")
            im_dark.save("training_patches_brightnessdown/" +
                         folder + "/" + filename + "_bdown" + ".png")


def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


def combineAllSteps(directory, folder):

    directory = directory+folder

    for file in os.scandir(directory):
        filepath = os.fsdecode(file)
        pathname, extension = os.path.splitext(filepath)

        if(filepath.endswith(".jpg") or filepath.endswith(".png")):
            img = Image.open(filepath)
            # image brightness enhancer
            enhancer = ImageEnhance.Brightness(img)

            factor = 1.2  # brightens the image
            im_bright = enhancer.enhance(factor)
            img_180_bright = im_bright.rotate(180, expand=0)
            img_180_bright_zmd = zoom_at(img_180_bright, 128, 128, 1.2)

            # left25
            left = 25
            top = 0
            right = 256
            bottom = 256
            im1 = img_180_bright_zmd.crop((left, top, right, bottom))
            im2 = Image.new('RGB', (25, 256))
            comb = Image.new('RGB', (im1.width + im2.width, im1.height))
            comb.paste(im1, (0, 0))
            comb.paste(im2, (im1.width, 0))

            filename = os.path.split(filepath)[1].split(".")[0]

            comb.save("training_patches_combined/" + folder +
                      "/" + filename + "_comb" + ".png")


def performDataAugmentation(directory: str, categories: list[str], augmentations: list[str]):
    '''
    Performs various data augmentation steps and creates a corresponding folder structure.

    @directory - The folder which contains the data that should be augmented.\n
    @categories - The classes for which the augmentations should be performed.\n\t\t Valid values: 'background', 'ponds', 'pools', 'solar' or 'trampoline'. \n
    @augmentations - The augmentations that should be performed.\n\t\t Valid values: 'rotate_images', 'move_images', 'zoom_images', 'change_brightness', 'combine_augmentations'. 
    '''
    createFolderStructure()
    for category in categories:

        if("rotate_images" in augmentations):
            rotateDirectory(directory, category)

        if("move_images" in augmentations):
            moveImageContentForDirectory(directory, category)

        if("zoom_images" in augmentations):
            zoomImagesForDirectory(directory, category)

        if("change_brightness" in augmentations):
            changeBrightnessForDirectory(directory, category)

        if("combine_augmentations" in augmentations):
            combineAllSteps(directory, category)


def createFolder(dirName: str):
    # Create target directory & all intermediate directories if don't exists
    try:
        os.makedirs(dirName)
        print("Directory ", dirName,  " Created ")
    except FileExistsError:
        print("Directory ", dirName,  " already exists")

    # Create target directory & all intermediate directories if don't exists
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName,  " Created ")
    else:
        print("Directory ", dirName,  " already exists")


def createFolderStructure():
    new_directories = [
        'training_patches_brightnessdown',
        'training_patches_brightnessup',
        'training_patches_combined',
        'training_patches_down',
        'training_patches_up',
        'training_patches_right',
        'training_patches_left',
        'training_patches_rotation',
        'training_patches_zoom']
    subfolders = ["background", "solar", "ponds", "trampoline", "pools"]

    for d in new_directories:
        for folder in subfolders:
            path = os.path.join(d, folder)
            createFolder(path)
