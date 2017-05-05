#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6.1 -*-

import os
from sys import platform

if platform == "darwin":
    import matplotlib as mil
    mil.use('TkAgg') # Fix plt RTE under macOS

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from stLogging import stLogger

_TRAINSET_PATH = ["./pics_ann/train/"]
_TESTSET_PATH = ["./pics_ann/test/"]
_TAG_PATH = "./pics_ann/tag_result/"
_OUTPUT_PERFIX = "./image_out/"

iblog = stLogger("stImage.log", loglevel = 30).log

class ImageBox():
    VAR = None

    def get_covbox_size(self, xbox, ybox, dpi = 300):
        sum = 0.0
        for i in range(0, len(xbox) - 1):
            sum += abs(xbox[i] * ybox[i + 1] - xbox[i + 1] * ybox[i])
        return sum / 2 / dpi

    def read_tag(self, picname):
        xbox = []
        ybox = []
        content = []
        with open(_TAG_PATH + picname + ".txt", "r") as t:
            text = ' '.join(t.readline().split()).split(";")
            assert(text[0].replace("\t", "").split(" ")[0] == picname)
            text[0] = text[0][text[0].find(" "):-1]
            for line in text:
                line = line.strip().split(" ")
                xbox.append(list(map(int, line[0:-2:2])))
                ybox.append(list(map(int, line[1:-1:2])))
                content.append(line[-1])
        return np.array(xbox), np.array(ybox), content

    def show_with_tag(self, picname, picpath = _TRAINSET_PATH[0], ec = 'y'):
        (xbox, ybox, content) = self.read_tag(picname)
        im = np.array(Image.open(picpath + picname), dtype=np.uint8)
        fig, ax = plt.subplots(1)
        ax.imshow(im)
        xy = np.array([xbox.T, ybox.T]).T
        for i in (xy):
            ax.add_patch(patches.Polygon(i, fill = False, edgecolor = ec))
        plt.show()

    def crop_to_box(self, img, box):
        return img.crop(map(int, box))

    def scale_to_fit(self, img, maxLength = 256):
        if (img.size[0] > img.size[1]):
            box = (maxLength, int(img.size[1] * maxLength / img.size[0]))
        else:
            box = (int(img.size[0] * maxLength / img.size[1]), maxLength)
        return img.resize(box)

    def scale_by_ratio(self, img, ratio = 1.0):
        return img.resize(tuple(map(int, (np.array(img.size) * ratio))))

def init():
    for dpath in (_TRAINSET_PATH + _TESTSET_PATH):
        if not os.path.isdir(dpath):
            iblog.error("Data path not exist: " + str(dpath))
        else:
            if not os.path.isdir(_OUTPUT_PERFIX + dpath):
                os.makedirs(_OUTPUT_PERFIX + dpath)

if __name__ == "__main__":
    init()
    for dpath in _TRAINSET_PATH:
        for file in os.listdir(dpath):
            pass
