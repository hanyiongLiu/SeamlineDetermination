import os
import cv2 as cv
import time
from support.myExpPack import StitchImages
from support.myExpPack import SearchSeamline
from support.myExpPack import ImageTrans
from TrainAndPredict import Predict
from support.myExpPack import ImageTrans


def mosaicking_image(img1_name, img2_name, img2_mask):
    tm = time.time()
    tm1 = time.time()
    img2_name = img2_name
    # img2_name = 'test1.png'
    img1_name = img1_name
    # img1_name = 'test2.png'

    # img_dir
    images_dir = 'images'
    masks_dir = 'masks2'
    results_dir = 'results/exp1'

    # img1 = cv.imread('images/DSC00714.JPG', 0)
    # img2 = cv.imread('images/DSC00715.JPG', 0)
    # img1_multi = cv.imread('images/DSC00714.JPG', 1)
    # img2_multi = cv.imread('images/DSC00715.JPG', 1)
    # img1_mask = cv.imread('masks2/mask_DSC00714.png', 0)
    # img2_mask = cv.imread('masks2/mask_DSC00715.png', 0)
    img1 = cv.imread(os.path.join(images_dir, img1_name), 0)
    img2 = cv.imread(os.path.join(images_dir, img2_name), 0)
    img1_multi = cv.imread(os.path.join(images_dir, img1_name), 1)
    img2_multi = cv.imread(os.path.join(images_dir, img2_name), 1)
    img1_mask = cv.imread(
        os.path.join(
            masks_dir,
            'mask_' +
            img1_name.split('.')[0] +
            '.png'),
        0)
    # img2_mask = cv.imread(
    #     os.path.join(
    #         masks_dir,
    #         'mask_' +
    #         img2_name.split('.')[0] +
    #         '.png'),
    #     0)
    img2_mask = img2_mask
    print('importing image consumed:', round(time.time() - tm1, 2), 's')
    tm1 = time.time()

    img1_corner_points, img2_corner_points, img1_gray_tran, img2_gray_trans, img1_mask, img2_mask, M1, M2 = ImageTrans(
        img1, img2, img1_mask, img2_mask)()
    print('transforming images consumed:', round(time.time() - tm1, 2), 's')
    tm1 = time.time()
    seamline, intersection_points = SearchSeamline(
        img1_corner_points,
        img2_corner_points,
        img1_gray_tran,
        img2_gray_trans,
        img1_mask,
        img2_mask, downsample_times=4)()
    print('searching seamline consumed:', round(time.time() - tm1, 2), 's')
    tm1 = time.time()

    stitch_image = StitchImages(img1_gray_tran,
                                img2_gray_trans,
                                img1_multi,
                                img2_multi,
                                img1_corner_points,
                                img2_corner_points,
                                seamline,
                                M1,
                                M2,
                                intersection_points)()
    print('stitching images consumed:', round(time.time() - tm1, 2), 's')
    tm1 = time.time()
    # cv.imwrite(
    #     os.path.join(results_dir, '{}_{}.{}'.format(
    #         img1_name.split('.')[0],
    #         img2_name.split('.')[0],
    #         img2_name.split('.')[1])),
    #     stitch_image)
    cv.imwrite(os.path.join(images_dir, 'out.jpg'), stitch_image)
    print('imwrite consumed:', round(time.time() - tm1, 2), 's')

    print('end, total consumed:', round(time.time() - tm, 2), 's')


class ImageMosaicking(object):
    def __init__(self, img1_name, img2_name, img2_mask):
        self.img1_name = img1_name
        self.img2_name = img2_name
        self.img2_mask = img2_mask

    def __call__(self):
        mosaicking_image(self.img1_name, self.img2_name, self.img2_mask)


if __name__ == '__main__':
    # dir = os.path.join('images')
    # img_name = []
    # for _, _, files in os.walk(dir):
    #     print(os.walk(dir))
    #     for file in files:
    #         if file.split('.')[-1] == 'JPG' and file[0] == 'D':
    #             img_name.append(file)
    # print(img_name)

    img_name = ['DSC00720.JPG', 'DSC00721.JPG', 'DSC00722.JPG', 'DSC00723.JPG', 'DSC00724.JPG', 'DSC00725.JPG']
    img1_name = 'DSC00720.JPG'
    img2_name = 'DSC00721.JPG'
    img2_mask = cv.imread('masks2/mask_DSC00721.png', 0)
    predictor = Predict()
    for i in range(len(img_name) - 1):
        print(f'{i + 1}/{len(img_name) - 1}:')
        ImageMosaicking(img1_name, img2_name, img2_mask)()
        try:
            img1_name = img_name[i + 2]
            img2_name = 'out.JPG'
            predictor('images/out.JPG')
            img2_mask = cv.imread('masks2/out.png', 0)
        except:
            pass
        # mosaicking_image(img1_name, img2_name, img2_mask)
