import os
import time

import cv2 as cv
import numpy as np
from TrainAndPredict import Predict
from support.img_trans import ImageTrans
from support.myExpPac_1126 import SearchSeamline
from support.myExpPac_1126 import ImageProcess
from support.myExpPac_1126 import Dijkstra


def get_image_names():
    img_dir = 'images/'
    dir = os.path.join('images')
    img_names = []
    img_names_plus = []
    img_fixed_name = []
    img_mask_fixed_name = []
    for _, _, files in os.walk(dir):
        print(os.walk(dir))
        for file in files:
            if file.split('.')[-1] == 'tif' and file[0:3] == '825':
                img_names.append(file)
                img_names_plus.append(img_dir + file)
                img_fixed_name.append('images/fixed_' + file)
                # img_mask_fixed_name.append('mask2/fixed_mask_' + file)
                img_mask_fixed_name.append('roadmask/fixed_mask_' + file)
    print(img_names)
    # for i in range(7):
    #     img_names.append('DSC007{}.JPG'.format(i + 20))
    #     img_names_plus.append('images/DSC007{}.JPG'.format(i + 20))
    #     img_fixed_name.append('images/fixed_DSC007{}.JPG'.format(i + 20))
    #     img_mask_fixed_name.append('mask2/fixed_mask_DSC007{}.JPG'.format(i + 20))
    return img_names, img_names_plus, img_fixed_name, img_mask_fixed_name


def get_mask(img_names):
    mask_dir = 'mask2/'
    image_dir = 'images/'
    predictor = Predict()
    for img_name in img_names:
        mask = predictor(image_dir + img_name)
        cv.imwrite(mask_dir + 'mask_' + img_name[:-4] + '.png', mask)


def _find_contours(img_name):
    img = cv.imread(img_name, 0)
    img[img < 10] = 0
    img[img != 0] = 255
    img_contours, _ = cv.findContours(
        img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    img_contours = [tuple(x) for x in img_contours[0].reshape(-1, 2)]
    return img_contours


def search_seamline(img_name):
    # sorted_images=[]
    # for ip in sorted(intersection_points):
    #     sorted_images.append(intersection_points[ip][0])
    intersection_points = []
    img1_name = img_name[0]
    for i in range(len(img_name) - 1):
        img2_name = img_name[i + 1]
        img1_contours = _find_contours(img1_name)
        img2_contours = _find_contours(img2_name)
        # = list(set(img1_contours).intersection(img2_contours))


class Mosaicking(object):
    def __init__(
            self,
            img1_name,
            img2_name,
            building_mask,
            downsample_times=5):
        self.img1_name = img1_name
        self.img2_name = img2_name
        self.building_mask = building_mask
        self.downsample_times = downsample_times

        self.img1 = cv.imread(img1_name)
        self.img2 = cv.imread(img2_name)
        self.img1_contours = self._find_contours(self.img1)
        self.img2_contours = self._find_contours(self.img2)
        self.intersection_pts = self.get_intersection_points()
        self.intersection_pts_in_img = self._get_intersection_points_in_img()
        self.seam_line = self._get_seamline()

    def __call__(self):
        img1_stitch_mask = self._get_stitch_mask(self.img2, self.img1_contours)
        img2_stitch_mask = self._get_stitch_mask(self.img1, self.img2_contours)
        img1_stitch_mask = np.array(
            [img1_stitch_mask for _ in range(3)], dtype=np.uint8).transpose(1, 2, 0)
        img2_stitch_mask = np.array(
            [img2_stitch_mask for _ in range(3)], dtype=np.uint8).transpose(1, 2, 0)
        img1_multi_masked = cv.multiply(self.img1, img1_stitch_mask)
        img2_multi_masked = cv.multiply(self.img2, img2_stitch_mask)
        stitched_image = cv.add(img1_multi_masked, img2_multi_masked)

        i_num=0
        for i in range(len(self.seam_line) - 1):
            cv.line(stitched_image, self.seam_line[i], self.seam_line[i + 1], (0, 255, 255), 10)
            if i % 20 == 0:
                cv.line(stitched_image, self.seam_line[i], self.seam_line[i], (255, 0, 0), 10)
            if i % 500 == 0:
                i_num+=1;
                cv.line(stitched_image, self.seam_line[i], self.seam_line[i], (0, 0, 255), 10)
        print("观测点个数：{}".format(i_num))

        # with open('seamline.txt', 'a') as file:
        #     file.write(str(self.seam_line)+'.')
        # with open('seamline.txt', 'r') as file:
        #     contents = file.read()
        # content = contents.split('.')
        #
        # color=[(250,250,255),(139,134,0)]
        # for j in range(len(content)-1):
        #     seam = eval(content[j])
        #     for i in range(len(seam)-1):
        #         cv.line(stitched_image, seam[i], seam[i + 1], color[j], 20)
        return stitched_image


    def get_intersection_points(self):
        intersection_points = list(
            set(self.img1_contours).intersection(self.img2_contours))
        ip_tm = [np.array(x) for x in intersection_points]
        n = len(intersection_points)
        max_dis = 0.0
        intersection_pts = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                dis = np.sqrt(
                    np.sum(
                        (ip_tm[i] -
                         ip_tm[j]) ** 2))
                if dis > max_dis:
                    max_dis = dis
                    intersection_pts = [
                        intersection_points[i], intersection_points[j]]
        return intersection_pts


    def _find_contours(self, img):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        img[img < 10] = 0
        img[img != 0] = 255
        img_contours, _ = cv.findContours(
            img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        img_contours = [tuple(x) for x in img_contours[0].reshape(-1, 2)]
        return img_contours


    def _get_seamline(self):
        print(self.intersection_pts)
        times = self.downsample_times
        impro = ImageProcess(times)
        # cost_im_downsample = impro.down_sample_2x2(self._get_cost_image())
        cost_im_downsample = impro.down_sample_2x2(self._get_cost_image_road())
        print('cost image size:', cost_im_downsample[0].shape)
        seed_downsample = impro.intersection_points_in_im(
            self.intersection_pts_in_img[0])
        end_downsample = impro.intersection_points_in_im(
            self.intersection_pts_in_img[1])

        path_in_img = [None] * (times + 1)
        for i in range(times, 0, -1):
            tm = time.time()
            dijkstra = Dijkstra(cost_im_downsample[i])

            path_in_img[i] = dijkstra(seed_downsample[i], end_downsample[i])

            cost_im_downsample[i - 1] = impro.path_upsample(
                path_in_img[i], cost_im_downsample[i - 1])
            print('feature {} consumed:'.format(i),
                  round(time.time() - tm, 2), 's')
        tm = time.time()
        dijkstra = Dijkstra(cost_im_downsample[0])
        path_in_img[0] = dijkstra(seed_downsample[0], end_downsample[0])
        print('feature 0 consumed:', round(time.time() - tm, 2), 's')

        left = min([x[0] for x in self.intersection_pts])
        top = min([x[1] for x in self.intersection_pts])
        path = [(x[0] + left, x[1] + top) for x in path_in_img[0]]
        return path


    def _get_cost_image(self, kernel_size=2):
        left_image, right_image = self._get_overlapping_images()

        # cv.imwrite('left_overlapping_region.JPG', left_image)
        # cv.imwrite('right_overlapping_region.JPG', right_image)

        assert left_image.shape == right_image.shape, 'the shapes of overlapping images are not same'

        left_image = left_image.astype(np.float32)
        right_image = right_image.astype(np.float32)
        cost_image = np.zeros_like(left_image[:, :, 0])
        for i in range(left_image.shape[2]):
            left_image_i = left_image[:, :, i]
            right_image_i = right_image[:, :, i]
            kernel = np.ones((kernel_size, kernel_size))
            dividend1 = cv.filter2D(left_image_i * right_image_i, -1,
                                    kernel, borderType=cv.BORDER_CONSTANT)
            dividend2 = (cv.filter2D(left_image_i, -1, kernel, borderType=cv.BORDER_CONSTANT)
                         * cv.filter2D(right_image_i, -1, kernel, borderType=cv.BORDER_CONSTANT)) / 25

            divisor1 = cv.filter2D(left_image_i * left_image_i, -1,
                                   kernel, borderType=cv.BORDER_CONSTANT)
            divisor2 = (cv.filter2D(left_image_i, -1, kernel,
                                    borderType=cv.BORDER_CONSTANT) ** 2) / 25
            divisor3 = cv.filter2D(right_image_i * right_image_i, -1,
                                   kernel, borderType=cv.BORDER_CONSTANT)
            divisor4 = (cv.filter2D(right_image_i, -1, kernel,
                                    borderType=cv.BORDER_CONSTANT) ** 2) / 25

            qncc = (dividend1 - dividend2) / \
                   np.sqrt((divisor1 - divisor2) * (divisor3 - divisor4))
            cost_image += 0.5 - 0.5 * qncc
        cost_image /= left_image.shape[2]

        cost_image *= 5.5
        cost_image = np.power(np.e, cost_image)

        # cost_image[0:2, :] = 0
        # cost_image[-2:, :] = 0
        # cost_image[:, 0:2] = 0
        # cost_image[:, -2:] = 0
        # cost_image[np.isnan(cost_image)] = 0
        #
        # tmp_list = np.sort(cost_image.reshape(-1))
        # threshold = tmp_list[-5 * (left_image.shape[0] + left_image.shape[1])]
        # cost_image[cost_image > threshold] = 0
        # cost_image[cost_image < 0] = 0
        #
        # cost_image *= int(5 / threshold)
        # #
        # # # cost_image *= 2000
        # cost_image = np.power(np.e, cost_image)
        cost_image[np.isnan(cost_image)] = 255
        # #
        # cost_image[cost_image == 1] = 255
        #
        overlapping1_mask, overlapping2_mask = self._get_overlapping_images(
            mask_mode=True)

        cost_image = self.mask_modify(cost_image, overlapping1_mask, 100)
        cost_image = self.mask_modify(cost_image, overlapping2_mask, 100)

        cost_image[0:2, :] = 255
        cost_image[-2:, :] = 255
        cost_image[:, 0:2] = 255
        cost_image[:, -2:] = 255

        tmp_cost_map = cost_image
        tmp_cost_map[tmp_cost_map < 100] *= 20
        cv.imwrite('cost_map.jpg', tmp_cost_map)
        return cost_image


    def _get_cost_image_road(self, kernel_size=2):
        left_image, right_image = self._get_overlapping_images()

        # cv.imwrite('left_overlapping_region.JPG', left_image)
        # cv.imwrite('right_overlapping_region.JPG', right_image)

        assert left_image.shape == right_image.shape, 'the shapes of overlapping images are not same'

        left_image = left_image.astype(np.float32)
        right_image = right_image.astype(np.float32)
        cost_image = np.zeros_like(left_image[:, :, 0])
        for i in range(left_image.shape[2]):
            left_image_i = left_image[:, :, i]
            right_image_i = right_image[:, :, i]
            kernel = np.ones((kernel_size, kernel_size))
            dividend1 = cv.filter2D(left_image_i * right_image_i, -1,
                                    kernel, borderType=cv.BORDER_CONSTANT)
            dividend2 = (cv.filter2D(left_image_i, -1, kernel, borderType=cv.BORDER_CONSTANT)
                         * cv.filter2D(right_image_i, -1, kernel, borderType=cv.BORDER_CONSTANT)) / 25

            divisor1 = cv.filter2D(left_image_i * left_image_i, -1,
                                   kernel, borderType=cv.BORDER_CONSTANT)
            divisor2 = (cv.filter2D(left_image_i, -1, kernel,
                                    borderType=cv.BORDER_CONSTANT) ** 2) / 25
            divisor3 = cv.filter2D(right_image_i * right_image_i, -1,
                                   kernel, borderType=cv.BORDER_CONSTANT)
            divisor4 = (cv.filter2D(right_image_i, -1, kernel,
                                    borderType=cv.BORDER_CONSTANT) ** 2) / 25

            qncc = (dividend1 - dividend2) / \
                   np.sqrt((divisor1 - divisor2) * (divisor3 - divisor4))
            cost_image += 0.5 - 0.5 * qncc
        cost_image /= left_image.shape[2]

        cost_image *= 100
        cost_image[np.isnan(cost_image)] = 255
        cost_image[0:2, :] = 255
        cost_image[-2:, :] = 255
        cost_image[:, 0:2] = 255
        cost_image[:, -2:] = 255
        overlapping1_mask, overlapping2_mask = self._get_overlapping_images(
            mask_mode=True)
        cost_image[np.isnan(cost_image)] = 255
        cost_image = self.mask_modify_road(cost_image, overlapping1_mask, 0.1)
        cost_image = self.mask_modify_road(cost_image, overlapping2_mask, 0.1)

        tmp_cost_map = cost_image

        cv.imwrite('cost_map.jpg', tmp_cost_map)
        return cost_image


    def _get_cost_image2(self, kernel_size=5):
        left_image, right_image = self._get_overlapping_images()
        assert left_image.shape == right_image.shape, 'the shapes of overlapping images are not same'

        left_image = left_image.astype(np.float32)
        right_image = right_image.astype(np.float32)
        cost_image = np.zeros_like(left_image[:, :, 0])
        for i in range(left_image.shape[2]):
            left_image_i = left_image[:, :, i]
            right_image_i = right_image[:, :, i]
            kernel = np.ones((kernel_size, kernel_size))
            dividend1 = cv.filter2D(left_image_i * right_image_i, -1,
                                    kernel, borderType=cv.BORDER_CONSTANT)
            dividend2 = (cv.filter2D(left_image_i, -1, kernel, borderType=cv.BORDER_CONSTANT)
                         * cv.filter2D(right_image_i, -1, kernel, borderType=cv.BORDER_CONSTANT)) / 25

            divisor1 = cv.filter2D(left_image_i * left_image_i, -1,
                                   kernel, borderType=cv.BORDER_CONSTANT)
            divisor2 = (cv.filter2D(left_image_i, -1, kernel,
                                    borderType=cv.BORDER_CONSTANT) ** 2) / 25
            divisor3 = cv.filter2D(right_image_i * right_image_i, -1,
                                   kernel, borderType=cv.BORDER_CONSTANT)
            divisor4 = (cv.filter2D(right_image_i, -1, kernel,
                                    borderType=cv.BORDER_CONSTANT) ** 2) / 25

            qncc = (dividend1 - dividend2) / \
                   np.sqrt((divisor1 - divisor2) * (divisor3 - divisor4))
            cost_image += 0.5 - 0.5 * qncc
        cost_image /= left_image.shape[2]

        cost_image[np.isnan(cost_image)] = 1

        overlapping1_mask, overlapping2_mask = self._get_overlapping_images(
            mask_mode=True)

        cost_image = self.mask_modify(cost_image, overlapping1_mask, 100)
        cost_image = self.mask_modify(cost_image, overlapping2_mask, 100)

        cost_image[0:2, :] = 1
        cost_image[-2:, :] = 1
        cost_image[:, 0:2] = 1
        cost_image[:, -2:] = 1
        cost_image[0:2, 0:2] = 0
        cost_image[-2:, 0:2] = 0
        cost_image[0:2, -2:] = 0
        cost_image[-2:, -2:] = 0
        return cost_image


    def _get_overlapping_images(self, mask_mode=False):
        left = min([x[0] for x in self.intersection_pts])
        right = max([x[0] for x in self.intersection_pts])
        top = min([x[1] for x in self.intersection_pts])
        bottom = max([x[1] for x in self.intersection_pts])
        if bottom - top < 300 and bottom - top < 0.1 * self.img1.shape[0]:
            top -= 200
            bottom += 200
        if right - left < 300 and right - left < 0.1 * self.img1.shape[1]:
            left -= 200
            right += 200
        overlapping1 = self.building_mask[top:bottom +
                                              1, left:right +
                                                      1] if mask_mode else self.img1[top:bottom +
                                                                                         1, left:right +
                                                                                                 1, :]
        overlapping2 = self.building_mask[top:bottom +
                                              1, left:right +
                                                      1] if mask_mode else self.img2[top:bottom +
                                                                                         1, left:right +
                                                                                                 1, :]
        return overlapping1, overlapping2


    def _get_intersection_points_in_img(self):
        left = min([x[0] for x in self.intersection_pts])
        top = min([x[1] for x in self.intersection_pts])
        intersection_pts_in_im = [(x[0] - left, x[1] - top)
                                  for x in self.intersection_pts]
        return intersection_pts_in_im


    def _get_stitch_mask(self, img_another, img_contours):
        point1_index = img_contours.index(self.intersection_pts[0])
        point2_index = img_contours.index(self.intersection_pts[1])
        if point1_index > point2_index:
            point1_index, point2_index = point2_index, point1_index
        list1 = img_contours[point2_index:] + img_contours[:point1_index + 1]
        list2 = img_contours[point1_index:point2_index + 1]
        img_another = cv.cvtColor(img_another, cv.COLOR_RGB2GRAY)
        img_another[img_another < 10] = 0
        img_another[img_another != 0] = 1
        list1_probability = [x for x in list1 if img_another[x[1], x[0]] == 1]
        list2_probability = [x for x in list2 if img_another[x[1], x[0]] == 1]
        list = list1 if len(list1_probability) < len(
            list2_probability) else list2
        contour = list + self.seam_line if self._distance(list[-1], self.seam_line[0]) \
                                           < self._distance(list[-1], self.seam_line[-1]) \
            else list + self.seam_line[::-1]
        mask = np.zeros_like(img_another)
        mask = cv.polylines(mask, [np.int32(contour)], True, 1)
        mask = cv.fillPoly(mask, np.int32([contour]), 1)
        return mask


    def _distance(self, point1, point2):
        res = np.sqrt(np.sum(
            (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
        ))
        return res


    def mask_modify(self, cost_im, mask, add_num):
        cost_im[mask != 0] += add_num
        cost_im[cost_im > 255] = 255
        cost_im = np.array(cost_im, dtype=np.float32)
        return cost_im


    def mask_modify_road(self, cost_im, mask, coefficient):
        cost_im[mask == 255] *= coefficient
        cost_im = np.array(cost_im, dtype=np.float32)
        return cost_im


def get_building_mask(img_mask_fixed_name):
    mask = cv.imread(img_mask_fixed_name[0], 0)
    for i in range(len(img_mask_fixed_name) - 1):
        tmp = cv.imread(img_mask_fixed_name[i + 1], 0)
        mask = cv.add(mask, tmp)
    # mask[mask < 100] = 0
    # mask[mask > 100] = 255
    mask[mask > 0] = 255
    return mask


if __name__ == '__main__':
    tm = time.time()
    img_names, img_names_plus, img_fixed_name, img_mask_fixed_name = get_image_names()
    # get_mask(img_names)
    ImageTrans(img_names_plus)()
    building_mask = get_building_mask(img_mask_fixed_name)
    # cv.imwrite('building_mask.png', building_mask)
    cv.imwrite('road_mask.png', building_mask)

    img1 = img_fixed_name[0]
    img2 = img_fixed_name[1]
    for i in range(len(img_fixed_name) - 1):
        print(f'{i + 1}/{len(img_fixed_name) - 1}')
        stitched_image = Mosaicking(
            img1,
            img2,
            building_mask)()
        cv.imwrite('tmp{}.JPG'.format(i), stitched_image)

        try:
            img1 = 'tmp{}.JPG'.format(i)
            img2 = img_fixed_name[i + 2]
        except:
            pass
    print(round(time.time() - tm, 2), 's', sep='')
