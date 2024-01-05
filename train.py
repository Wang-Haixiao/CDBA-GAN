# -*- coding: utf-8 -*-
from config import Config
from SAGAN_conditional import SAGAN
import os
import cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
if __name__ == "__main__":
    config = Config()
    model = SAGAN(config)
    if config.RESUME_TRAIN:
        discriminator_path = "./result/181104_1905/weights/ramen_cam/discriminator245000.hdf5"
        generator_path = "./result/181104_1905/weights/ramen_cam/generator245000.hdf5"
        print("Training start at {} iterationNones".format(config.COUNTER))
        model.resume_train(discriminator_path, generator_path, config.COUNTER)
    else:
        print("Training start")
        #tt#
        # sonar_path="./data/sonar/"

        #change .jpg
        # for filename in os.listdir(sonar_path):
        #     position = os.path.splitext(filename)
        #     if position[1]==".jpg":
        #         newname = position[0]+".png"
        #         # os.chdir(sonar_path)
        #         filename = sonar_path + filename
        #         newname = sonar_path + newname
        #         os.rename(filename, newname)
        #
        #resize
        # for filename in os.listdir(sonar_path):
        #     image = cv2.imread(fi  lename,0)
        #     resize_image = cv2.resize(image ,(512,400),interpolation=cv2.INTER_CUBIC)
        #     cv2.imwrite(filename, resize_image)
        # #tt#
        model.train()
        # model.generate(3, config.MODEL_FILE_PATH, 100)