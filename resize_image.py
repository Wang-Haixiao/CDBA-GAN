import os
from PIL import Image
import cv2
def produceImage(filename, width, height):
    # image = Image.open(sonar_path + filename)
    # resized_image = image.resize((width, height), Image.ANTIALIAS)
    # resized_image.save(sonar_path + filename)
    image = cv2.imread(sonar_path + filename)
    resized_image = cv2.resize(image,(width, height))
    cv2.imwrite((sonar_path + filename), resized_image)
if __name__=='__main__':
    sonar_path = "./data/4/"
    width = 192
    height = 256
    for filename in os.listdir(sonar_path):
        try:
            produceImage(filename, width, height)
        except(OSError, NameError ):
            print('%s' %filename)
        print('done')