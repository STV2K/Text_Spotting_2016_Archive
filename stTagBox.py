#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6.1 -*-

import os
import random
import numpy as np
from PIL import Image
from sys import platform

if platform == "darwin":
    import matplotlib as mil
    mil.use('TkAgg') # Fix RTE caused by pyplot under macOS

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import matplotlib.path as mplPath

_OLD_TAG_PATH = "./pics_ann/tag_result/"
_TRAINSET_PATH = ["./stv2k_train/"]
_TESTSET_PATH = ["./stv2k_test/"]
_OUTPUT_PERFIX = "./image_out/"

__TEST_PICNAME = ["STV2K_tr_1203.jpg",
                  "STV2K_tr_1206.jpg",
                  "STV2K_tr_1101.jpg",
                  "STV2K_tr_0010.jpg",
                  "STV2K_tr_0220.jpg"]

TAG_SHRINK = 4

class TagBox():

    def load_tag_old(self, picname):
        xbox = []
        ybox = []
        content = []
        with open(_OLD_TAG_PATH + picname + ".txt", "r") as t:
            text = ' '.join(t.readline().split()).split(";")
            assert(text[0].replace("\t", "").split(" ")[0] == picname)
            text[0] = text[0][text[0].find(" "):-1]
            for line in text:
                line = line.strip().split(" ")
                xbox.append(list(map(int, line[0:-2:2])))
                ybox.append(list(map(int, line[1:-1:2])))
                content.append(line[-1])
        return np.array(xbox), np.array(ybox), content

    def load_tag(self, picname, picpath):
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
        return (np.array(xbox).astype(int), np.array(ybox).astype(int), content)

    def shrink_tag(self, tag, ratio = 1.0):
        return (np.floor(tag[0] * ratio).astype(int), np.floor(tag[1] * ratio).astype(int), tag[2])

    def tag2array(self, tag, array_size):
        (xbox, ybox, content) = tag
        array = np.zeros(np.array(array_size), dtype=np.uint8).T
        xy = np.array([xbox.T, ybox.T]).T
        for i in range(0, len(xy)):
            maxx = max(xbox[i])
            maxy = max(ybox[i])
            minx = min(xbox[i])
            miny = min(ybox[i])
            # print([len(array), len(array[0])])
            if (maxx > array_size[0]):
                maxx = array_size[0]
            if (maxy > array_size[1]):
                maxy = array_size[1]
            if (minx < 0):
                minx = 0
            if (miny < 0):
                miny = 0
            # print([minx, maxx, miny, maxy])
            pth = mplPath.Path(xy[i])
            for x in range(minx, maxx):
                for y in range(miny, maxy):
                    if (pth.contains_point((x, y))):
                        # print([y, x])
                        array[y][x] = 1
        return array.flatten()

    def get_tag_array(self, tagfile, filepath, array_size):
        tag = self.load_tag(tagfile, filepath)
        return self.tag2array(self.shrink_tag(tag, 1 / TAG_SHRINK), array_size)

    def show_saliency_map(self, saliency_array, exp_size = (48, 64)):
        # May need normalize here
        saliency_array = np.array(saliency_array, dtype = np.float)
        maxv = max(saliency_array)
        minv = min(saliency_array)
        saliency_array += - minv
        saliency_array /= maxv - minv
        slcim = Image.fromarray((saliency_array * 255).astype(np.uint8).reshape(exp_size)).convert("L")
        slcim.show()

class MiniImageBox():

    def show_img(self, img):
        img.show()

    # def show_img_with_tag(self, img, tag, ec = 'y', cm = "Greys_r"):
    #     (xbox, ybox, content) = tag
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(img, cmap = cm)
    #     xy = np.array([xbox.T, ybox.T]).T
    #     for i in (xy):
    #         ax.add_patch(patches.Polygon(i, fill = False, edgecolor = ec))
    #     plt.show()

    def load_image(self, picname, picpath, mode = None):
        im = Image.open(picpath + picname)
        if (mode):
            # "1" for monochrome, "L" for greyscale, "LA" for "L" with alpha
            im = im.convert(mode)
        return im

    def image2nparray(self, img):
        return np.array(img, dtype = np.uint8)


tb = TagBox()
mib = MiniImageBox()

def get_data(picname, picpath):
    im = mib.load_image(picname, picpath, "L")
    # print(im.size)
    imgarr = mib.image2nparray(im).flatten()
    tagarr = tb.get_tag_array(picname, picpath, np.array(im.size) // TAG_SHRINK)
    return imgarr, tagarr

def walk_path(path, randpick = None):
    picset = []
    for rt, dr, fs in os.walk(path):
        for fn in fs:
            if (".jpg") in fn:
                picset.append(fn)
    if randpick:
        picset = random.sample(picset, randpick)
    return picset


if __name__ == "__main__":
    for picn in __TEST_PICNAME:
        imgarr, tagarr = get_data(picn, _TRAINSET_PATH[0])
        img = mib.load_image(picn, _TRAINSET_PATH[0])
        mib.show_img(img)
        # tag = tb.load_tag(picn, _TRAINSET_PATH[0])
        # mib.show_img_with_tag(img, tag)
        # slcarr = tagarr.reshape
        tb.show_saliency_map(tagarr, exp_size = (img.size[1] // TAG_SHRINK, img.size[0] // TAG_SHRINK))
