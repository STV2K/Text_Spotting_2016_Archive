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
_OLD_TAG_PATH = "./pics_ann/tag_result/"
_OUTPUT_PERFIX = "./image_out/"

__TEST_PICNAME = ["STV2K_tr_1203.jpg",
                  "STV2K_tr_1206.jpg",
                  "STV2K_tr_1101.jpg",
                  "STV2K_tr_0010.jpg",
                  "STV2K_tr_0220.jpg"]

iblog = stLogger("stImage", loglevel = 30).log

class ImageBox():

    def get_covbox_size(self, xpnts, ypnts, dpi = 1):
        sum = 0.0
        n = len(xpnts)
        for i in range(n):
            j = (i + 1) % n
            sum += xpnts[i] * ypnts[j] - xpnts[j] * ypnts[i]
        return abs(sum) / 2 / dpi

    def image2nparray(self, img):
        return np.array(img, dtype=np.uint8)

    def load_image(self, picname, picpath, mode = None):
        im = Image.open(picpath + picname)
        if (mode):
            # "1" for monochrome, "L" for greyscale, "LA" for "L" with alpha
            im = im.convert(mode)
        return im

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

    def save_tag(self, tag, filename, filepath):
        (xbox, ybox, content) = tag
        with open(_OUTPUT_PERFIX + filepath + filename, "w", encoding = "GBK") as f:
            for xps, yps, c in zip(xbox, ybox, content):
                first = True
                for i in range(len(xps)):
                    if (not first):
                        f.write(",")
                    first = False
                    f.write(str(xps[i]) + "," + str(yps[i]))
                    # print(str(xps[i]) + "," + str(yps[i]))
                f.write("\n" + c + "\n\n")

    def show_img_with_tag(self, img, tag, ec = 'y', cm = "Greys_r"):
        (xbox, ybox, content) = tag
        fig, ax = plt.subplots(1)
        ax.imshow(img, cmap = cm)
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

    def fill_area(self, img, tag, filter):
        (xbox, ybox, content) = tag
        imgn = self.image2nparray(img)
        xy = np.array([xbox.T, ybox.T]).T
        for i in range(0, len(xy)):
            if (filter[i]):
                maxx = max(xbox[i])
                maxy = max(ybox[i])         
                minx = min(xbox[i])
                miny = min(ybox[i])
                if (maxx > img.size[0]):
                    maxx = img.size[0]
                if (maxy > img.size[1]):
                    maxy = img.size[1]
                if (minx < 0):
                    minx = 0
                if (miny < 0):
                    miny = 0
                pth = mplPath.Path(xy[i])
                for x in range(minx, maxx):
                    for y in range(miny, maxy):
                        if (pth.contains_point((x, y))):
                            imgn[y][x] -= imgn[y][x]
                            # imgn[y][x] = [random.randrange(0, 256),
                            #             random.randrange(0, 256),
                            #             random.randrange(0, 256)]
        # self.show_img_with_tag(imgn, tag)
        return imgn

    def filter_gen(self, tag, threshold = 36.0):
        flt = [False] * len(tag[0])
        for i in range(0, len(tag[0])):
            if tag[2][i] == '':
                flt[i] = True # Remove the unrecognizable texts
            else:
                # print("Text: " + tag[2][i]+ " Len: " + str(len(tag[2][i])) + " AvgSize: " + str(self.get_covbox_size(tag[0][i], tag[1][i]) / len(tag[2][i])))
                # print("Xbox: " + str(tag[0][i]) +" Ybox: " + str(tag[1][i]))
                if ((self.get_covbox_size(tag[0][i], tag[1][i]) / len(tag[2][i])) < threshold):
                    flt[i] = True
        return flt

    def blank_tag_filter(self, tag, gen = False):
        (xbox, ybox, content) = tag
        flt = [False] * len(tag[0])
        for i in range(len(tag[0])):
            if (content[i] == ''):
                flt[i] = True
        if (gen):
            return flt
        else:
            return self.filter_tag(tag ,flt)

    def filter_croped_tag(self, tag, box):
        (xbox, ybox, content) = self.blank_tag_filter(tag) 
        cropth = mplPath.Path([[box[0], box[1]],
                            [box[0], box[3]],
                            [box[2], box[3]],
                            [box[2], box[1]]])
        xy = np.array([xbox.T, ybox.T]).T
        flt = [False] * len(xbox)
        for i in range(len(xy)):
            txtpth = mplPath.Path(xy[i])
            partial = False
            if (not cropth.contains_path(txtpth)):
                    for y in range(len(xbox[i])):
                        partial = partial or cropth.contains_point((xbox[i][y], ybox[i][y]))
                        if partial:
                            # print("Partial: " + content[i])
                            content[i] = ''
                            xbox[i] -= box[0]
                            ybox[i] -= box[1]
                            break                        
                    if not partial:
                        # print("Outside: " + content[i])
                        flt[i] = True
            if (not flt[i] and not partial):
                xbox[i] -= box[0]
                ybox[i] -= box[1]
                # print("Inside: " + content[i])
        return self.filter_tag((xbox, ybox, content), flt)

    def filter_tag(self, tag, filter):
        (xbox, ybox, content) = tag
        xboxn = []
        yboxn = []
        contn = []
        for i in range(len(filter)):
            if (not filter[i]):
                xboxn.append(xbox[i])
                yboxn.append(ybox[i])
                contn.append(content[i])
        return (np.array(xboxn).astype(int), np.array(yboxn).astype(int), contn)

    def divide_into_four(self, picname, picpath):
        img, tag = self.fitting_with_tag(self.load_image(picname, picpath, 'L'), self.load_tag(picname, picpath), 512)
        ori = img.size
        hhp = int(ori[0] / 2)
        hvp = int(ori[1] / 2)
        boxset = [(0, 0, hhp, hvp),
                (hhp, 0, ori[0], hvp),
                (0, hvp, hhp, ori[1]),
                (hhp, hvp, ori[0], ori[1])]
        img = Image.fromarray(self.fill_area(img, tag, self.blank_tag_filter(tag, True)))
        for i in range(0, len(boxset)):
            tagn = self.filter_croped_tag(tag, boxset[i])
            tagf = self.blank_tag_filter(tagn)
            if (len(tagf[0])):
                imn = self.fill_area(self.crop_to_box(img, boxset[i]), tagn, self.blank_tag_filter(tagn, True))
                Image.fromarray(imn).save(_OUTPUT_PERFIX + picpath + "crop" + str(i) + "-" + picname)
                self.save_tag(tagf, "crop" + str(i) + "-" + picname.replace(".jpg", ".txt"), picpath)

    def scale_with_tag(self, picname, picpath):
        img = self.load_image(picname, picpath, 'L')
        tag = self.load_tag(picname, picpath)
        shrimg, shrtag = self.fitting_with_tag(img, tag)
        shrimg = self.fill_area(shrimg, shrtag, self.filter_gen(shrtag))
        shrtag = self.filter_tag(shrtag, self.filter_gen(shrtag))
        if (len(shrtag[0])):
            Image.fromarray(shrimg).save(_OUTPUT_PERFIX + picpath + "scale-" + picname)
            self.save_tag(self.blank_tag_filter(shrtag), "scale-" + picname.replace(".jpg", ".txt"), picpath)

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
    #     img = ib.load_image(picn, _TRAINSET_PATH[0])
    #     tag = ib.load_tag(picn, _TRAINSET_PATH[0])
    #     ib.show_img_with_tag(img, tag)
    #     shrimg, shrtag = ib.fitting_with_tag(img, tag)
    #     ib.show_img_with_tag(shrimg, shrtag)
    # for picn in __TEST_PICNAME:
    #     img = ib.load_image(picn, _TRAINSET_PATH[0])
    #     tag = ib.load_tag(picn, _TRAINSET_PATH[0])
    #     shrimg, shrtag = ib.fitting_with_tag(img, tag)
    #     shrimg = ib.fill_area(shrimg, shrtag, ib.filter_gen(shrtag))
    #     ib.show_img_with_tag(shrimg, shrtag)
    # for picn in tqdm.tqdm(__TEST_PICNAME):
    #     ib.divide_into_four(picn, _TRAINSET_PATH[0])
    for dpath in [_TRAINSET_PATH[0], _TESTSET_PATH[0]]:
        for root, dirs, files in os.walk(dpath):
            picfs = list(filter(lambda x: ".jpg" in x, files))
            for fn in tqdm.tqdm(picfs):
                ib.divide_into_four(fn, dpath)
                ib.scale_with_tag(fn, dpath)
                