from PIL import Image
import cv2 as cv
import pandas as pd
import os


def saveOrPrintImages(path: str, print_to_output: bool = False, valBoundingBoxes: bool = False, thickness: int = 2, saveImagesPath: str = ""):
    '''
    Loads and saves all .jpg and .png files from the specified directory.\n

    @print - If set to True, will print the (annotated) images to the output (takes some time). Default value is False.

    @valBoundingBoxes If set to true, the method will load the bounding box data from the csv files. The default for this is False.\n

    @thickness parameter determines how thick the bounding boxes are drawn on the image (width of the line in pixels). The default value is 2.\n

    @saveImagesPath If specified will save the drawn images at the specified path.

    '''
    for file in os.scandir(path):
        filepath = os.fsdecode(file)

        if(not (filepath.endswith(".jpg") or filepath.endswith(".png")) or ("annotated" in filepath)):
            continue

        print("Saving / printing file: ",
              os.path.splitext(file.name)[0], "_annotated.jpg")
        image = cv.imread(filepath)

        if(valBoundingBoxes):
            image = addBoundingBoxesFromCsv(image, filepath, thickness)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Convert to PIL format
        image = Image.fromarray(image)
        if(saveImagesPath):
            image.save(saveImagesPath+"/" + os.path.splitext(file.name)
                       [0]+"_annotated.jpg", "JPEG")

        if(print_to_output):
            image.show()


def addBoundingBoxesFromCsv(image: cv.Mat, filepath: str, thickness: int) -> cv.Mat:
    '''
    Uses a csv file containing the predictions for a single image to draw bounding boxes into an image.\n

    Performs the changes in place and returns the modified image.
    The image parameter specifies the image to be modified.
    The filepath must contain the path of the csv file that contains the bounding box predictions for this image.
    '''
    csvPath = os.path.splitext(filepath)[0]+"_prediction.csv"
    boundingBoxesDataframe = pd.read_csv(csvPath, delimiter=",", header=0)

    for index, boundingBox in boundingBoxesDataframe.iterrows():
        label = boundingBox["label"]
        start_point = (int(boundingBox["x_upper_left"]), int(
            boundingBox["y_upper_left"]))
        end_point = (int(boundingBox["x_lower_right"]), int(
            boundingBox["y_lower_right"]))

        color: str

        if(label == "background"):
            color = (255, 255, 255)  # (BGR) White
        elif(label == "pool"):
            color = (255, 255, 0)  # (BGR) Cyan
        elif(label == "pond"):
            color = (0, 128, 0)  # (BGR) Green
        elif(label == "solar"):
            color = (255, 0, 0)  # (BGR) Blue
        elif(label == "trampoline"):
            color = (0, 255, 255)  # (BGR) Yellow
        else:
            color = (180, 105, 255)  # (BGR) Pink if no label present
        image = cv.rectangle(image, start_point, end_point,
                             color, thickness=thickness)

    return image
