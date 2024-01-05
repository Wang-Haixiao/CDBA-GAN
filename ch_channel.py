import os
import cv2
import skimage.io

if __name__=='__main__':
    sonar_path = "./data/sonar/"
    save_path = "./data/sonar_new/"
    for filename in os.listdir(sonar_path):
        image = cv2.imread(sonar_path + filename)
        # image = skimage.io.imread(sonar_path + filename)
        cv2.imwrite(save_path + filename, image)
#