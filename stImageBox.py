#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6.1 -*-

import os
import random
from sys import platform

if platform == "darwin":
    import matplotlib as mil
    mil.use('TkAgg') # Fix RTE caused by pyplot under macOS
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mplPath
import numpy as np
import tqdm
from PIL import Image

from stLogging import stLogger

_TRAINSET_PATH = ["./stv2k_train/"]
_TESTSET_PATH = ["./stv2k_test/"]
_TAG_PATH = "./pics_ann/tag_result/"
_OUTPUT_PERFIX = "./image_out/"

__TEST_PICNAME = ["STV2K_tr_1203.jpg",
                  "STV2K_tr_1206.jpg",
                  "STV2K_tr_1101.jpg",
                  "STV2K_tr_0010.jpg",
                  "STV2K_tr_0220.jpg"]

iblog = stLogger("stImage", loglevel = 30).log

class ImageBox():
    VAR = None

    def get_covbox_size(self, xpnts, ypnts, dpi = 300):
        sum = 0.0
        for i in range(0, len(xpnts) - 1):
            sum += abs(xpnts[i] * ypnts[i + 1] - xpnts[i + 1] * ypnts[i])
        return sum / 2 / dpi

    def image2nparray(self, img):
        return np.array(img, dtype=np.uint8)

    def load_image(self, picname, picpath = _TRAINSET_PATH[0], mode = None):
        im = Image.open(picpath + picname)
        if (mode):
            # "1" for monochrome, "L" for greyscale, "LA" for "L" with alpha
            im = im.convert(mode)
        return im

    def load_tag_old(self, picname):
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

    def load_tag(self, picname, picpath = _TRAINSET_PATH[0]):
        xbox = []
        ybox = []
        content = []
        with open(picpath + picname.replace(".jpg", ".txt"), "r", encoding = "GBK") as t:
            cnt = 0
            for line in t:
                if (cnt % 3 == 0 and line != ""):
                    line = line.strip().split(",")
                    xbox.append(list(map(int, line[0:-1:2])))
                    ybox.append(list(map(int, line[1::2])))
                if (cnt % 3 == 1):
                    content.append(line.strip())
                cnt += 1
        return (np.array(xbox), np.array(ybox), content)

    def shrink_tag(self, tag, ratio = 1.0):
        return (np.ceil(tag[0] * ratio).astype(int), np.floor(tag[1] * ratio).astype(int), tag[2])

    def show_img_with_tag(self, img, tag, ec = 'y'):
        (xbox, ybox, content) = tag
        fig, ax = plt.subplots(1)
        ax.imshow(img)
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

    def fill_area(self, img, tag, filter = None):
        (xbox, ybox, content) = tag
        imgn = self.image2nparray(img)
        xy = np.array([xbox.T, ybox.T]).T
        for i in range(0, len(xy)):
            if (content[i] == ''):# (filter[i]):
                maxx = max(xbox[i])
                maxy = max(ybox[i])         
                minx = min(xbox[i])
                miny = min(ybox[i])
                if (maxx > img.size[0]):
                    maxx = img.size[0]
                elif (maxy > img.size[1]):
                    maxy = img.size[1]
                pth = mplPath.Path(xy[i])
                for x in range(minx, maxx):
                    for y in range(miny, maxy):
                        if (pth.contains_point((x, y))):
                            imgn[y][x] -= imgn[y][x]
                            # imgn[y][x] = [0, 0, 0]
                            # imgn[y][x] = [random.randrange(0, 256),
                            #             random.randrange(0, 256),
                            #             random.randrange(0, 256)]
        self.show_img_with_tag(imgn, tag)
        return imgn, tag

    def fitting_with_tag(self, img, tag, maxLength = 256):
        sim = self.scale_to_fit(img, maxLength)
        return sim, self.shrink_tag(tag, ratio = sim.size[0] / img.size[0])

def init():
    for dpath in _TRAINSET_PATH + _TESTSET_PATH:
        if not os.path.isdir(dpath):
            iblog.error("Data path not exist: " + str(dpath))
        else:
            if not os.path.isdir(_OUTPUT_PERFIX + dpath):
                os.makedirs(_OUTPUT_PERFIX + dpath)

if __name__ == "__main__":
    init()
    ib = ImageBox()
    # for picn in __TEST_PICNAME:
    #     img = ib.load_image(picn)
    #     tag = ib.load_tag(picn)
    #     ib.show_img_with_tag(img, tag)
    #     shrimg, shrtag = ib.fitting_with_tag(img, tag)
    #     ib.show_img_with_tag(shrimg, shrtag)
    for picn in __TEST_PICNAME:
        img = ib.load_image(picn)
        tag = ib.load_tag(picn)
        shrimg, shrtag = ib.fitting_with_tag(img, tag)
        ib.fill_area(shrimg, shrtag)
