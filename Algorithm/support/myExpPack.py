import numpy as np
import cv2 as cv
import time
import math
from skimage.measure.block import block_reduce

from support.convex_polygon_intersection import _sort_vertices_anti_clockwise_and_remove_duplicates
from support.convex_polygon_intersection import _get_edge_intersection_points


class Dijkstra(object):

    def __init__(self, img):
        self.img = img
        self.h, self.w = self.img.shape

    def __call__(self, seed, end):
        return self.dijkstra_min_heap(seed, end)

    def get_neighbors(self, p):
        x, y = p  # 当前点坐标

        x_left = x - 1 if x != 0 else 0
        x_right = x + 1 if x != self.w - 1 else self.w - 1
        y_top = y - 1 if y != 0 else 0
        y_bottom = y + 1 if y != self.h - 1 else self.h - 1

        return [(x, y) for x in range(x_left, x_right + 1)
                for y in range(y_top, y_bottom + 1)]  # 范围3*3领域9个点坐标

    def neight_cost(self, next_p):
        return self.img[next_p[1]][next_p[0]]

    def item_search(self, cost, item):
        low = 0
        high = len(cost) - 1
        while low <= high:
            middle = (low + high) // 2
            if cost[middle][0] > item:
                high = middle - 1
            elif cost[middle][0] < item:
                low = middle + 1
            else:
                return middle
        return (high + low) // 2 + 1

    def small_path_point(self, seed, end, paths):
        path_piont = []
        path_piont.insert(0, end)  # 把结束点加到路径中
        while seed != end:  # 直到结束点坐标等于开始点坐标是结束
            top_point = paths[end]  # 更新的top_point为最短路径中某个点的上一个坐标点，即更加靠近种子点
            path_piont.append(top_point)  # 记录路径
            end = top_point  # 更新点坐标
        return path_piont

    def dijkstra_min_heap(self, seed, end):
        processMap = np.ones(self.img.shape, dtype=np.uint8)
        cost = [[255, seed]]
        path = {}
        while cost:
            p = cost[0][1]
            neighbors = self.get_neighbors(p)  # 当前成本代价最小值的领域节点
            processMap[p[1], p[0]] = 0
            for next_p in [
                x for x in neighbors if processMap[x[1], x[0]] != 0]:  # 没有被处理过的领域点坐标
                # 当前点与领域的点cost的差值 + 起始点到到当前点累计的cost值
                dik_cost = self.neight_cost(next_p) + cost[0][0]
                cost.insert(
                    self.item_search(
                        cost, dik_cost), [
                        dik_cost, next_p])  # 该领域所需代价值的更新
                processMap[next_p[1], next_p[0]] = 0
                path[next_p] = p  # 把cost最小点作为领域点next_p的前一个点

                if (next_p == end):  # 当前点到达结束点时，提前结束
                    cost = []  # 为了跳出循环
                    break  # 为了跳出循环
            if cost:
                cost.pop(0)  # 已经处理了的点就排除
        path = self.small_path_point(seed, end, path)
        return path


class StitchImages(object):
    def __init__(
            self,
            img1_gray,
            img2_gray,
            img1_multi,
            img2_multi,
            img1_pts,
            img2_pts,
            seamline,
            M1,
            M2,
            intersection_points):
        self.img1_gray = img1_gray
        self.img2_gray = img2_gray
        self.img1_multi = img1_multi
        self.img2_multi = img2_multi
        self.img1_pts = img1_pts
        self.img2_pts = img2_pts
        self.seamline = seamline
        self.M1 = M1
        self.M2 = M2
        self.intersection_points = intersection_points

    def __call__(self):
        mask1, mask2 = self._get_mask()
        img_out = self._get_mosaicked_image(mask1, mask2)
        return img_out

    def _concatenate_seamline_pts(self):
        im_pro = ImageProcess()
        img1_3pts = im_pro.get_most_points(
            self.intersection_points, self.img1_pts, self.img2_pts)
        img2_3pts = im_pro.get_most_points(
            self.intersection_points, self.img2_pts, self.img1_pts)
        polylines1 = self.seamline + img1_3pts
        polylines2 = self.seamline + img2_3pts
        return polylines1, polylines2

    def _get_mask(self):
        polylines1, polylines2 = self._concatenate_seamline_pts()
        assert self.img1_gray.shape == self.img2_gray.shape, "img1_gray and img2_gray don't have the same shape."
        mask1 = np.zeros(self.img1_gray.shape)
        mask2 = np.zeros(self.img1_gray.shape)
        mask1 = cv.polylines(mask1, [np.int32(polylines1)], True, 1)
        mask1 = cv.fillPoly(mask1, np.int32([polylines1]), 1)
        mask2 = cv.polylines(mask2, [np.int32(polylines2)], True, 1)
        mask2 = cv.fillPoly(mask2, np.int32([polylines2]), 1)
        return mask1, mask2

    def _get_mosaicked_image(self, mask1, mask2):
        h, w = self.img1_gray.shape
        img1_multi_trans = cv.warpPerspective(self.img1_multi, self.M1, (w, h))

        img2_multi_trans = cv.warpAffine(self.img2_multi, self.M2, (w, h))
        mask1 = np.array([mask1 for _ in range(3)],
                         dtype=np.uint8).transpose(1, 2, 0)
        mask2 = np.array([mask2 for _ in range(3)],
                         dtype=np.uint8).transpose(1, 2, 0)
        img1_multi_masked = cv.multiply(img1_multi_trans, mask1)
        img2_multi_masked = cv.multiply(img2_multi_trans, mask2)
        stitched_image = cv.add(img1_multi_masked, img2_multi_masked)
        return stitched_image


class SearchSeamline(object):

    def __init__(
            self,
            img1_corner_coordinate,
            img2_corner_coordinate,
            im1,
            im2,
            im1_mask,
            im2_mask,
            downsample_times=2):
        self.downsample_times = downsample_times
        self.img1_coordinate = img1_corner_coordinate
        self.img2_coordinate = img2_corner_coordinate
        self.img1 = im1
        self.img2 = im2
        self.img1_mask = im1_mask
        self.img2_mask = im2_mask

        self.intersection_pts = self._get_intersection_points()
        self.intersection_pts_in_img = self._get_intersection_points_in_img()

    def __call__(self):
        return self._get_seamline(), self.intersection_pts

    def _get_intersection_points(self):
        polygon1 = _sort_vertices_anti_clockwise_and_remove_duplicates(
            self.img1_coordinate)
        polygon2 = _sort_vertices_anti_clockwise_and_remove_duplicates(
            self.img2_coordinate)
        intersection_points = _get_edge_intersection_points(polygon1, polygon2)

        n = len(intersection_points)
        max_dis = 0.0
        intersection_pts = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                dis = np.sqrt(
                    np.sum(
                        (intersection_points[i] -
                         intersection_points[j]) ** 2))
                if dis > max_dis:
                    max_dis = dis
                    intersection_pts = [
                        tuple(
                            intersection_points[i].astype(
                                np.uint32)), tuple(
                            intersection_points[j].astype(
                                np.uint32))]
        return intersection_pts

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
        img1_tmp = self.img1_mask if mask_mode else self.img1
        img2_tmp = self.img2_mask if mask_mode else self.img2
        overlapping1 = img1_tmp[top:bottom + 1, left:right + 1]
        overlapping2 = img2_tmp[top:bottom + 1, left:right + 1]
        return overlapping1, overlapping2

    def _get_cost_image(self, kernel_size=5):
        left_image, right_image = self._get_overlapping_images()
        assert left_image.shape == right_image.shape, 'the shapes of overlapping images are not same'

        left_image = left_image.astype(np.float32)
        right_image = right_image.astype(np.float32)

        kernel = np.ones((kernel_size, kernel_size))
        dividend1 = cv.filter2D(left_image * right_image, -1,
                                kernel, borderType=cv.BORDER_CONSTANT)
        dividend2 = (cv.filter2D(left_image, -1, kernel, borderType=cv.BORDER_CONSTANT)
                     * cv.filter2D(right_image, -1, kernel, borderType=cv.BORDER_CONSTANT)) / 25

        divisor1 = cv.filter2D(left_image * left_image, -1,
                               kernel, borderType=cv.BORDER_CONSTANT)
        divisor2 = (cv.filter2D(left_image, -1, kernel,
                                borderType=cv.BORDER_CONSTANT) ** 2) / 25
        divisor3 = cv.filter2D(right_image * right_image, -1,
                               kernel, borderType=cv.BORDER_CONSTANT)
        divisor4 = (cv.filter2D(right_image, -1, kernel,
                                borderType=cv.BORDER_CONSTANT) ** 2) / 25

        qncc = (dividend1 - dividend2) / \
               np.sqrt((divisor1 - divisor2) * (divisor3 - divisor4))
        cost_image = 0.5 - 0.5 * qncc
        cost_image *= 20

        cost_image[np.isnan(cost_image)] = 255

        overlapping1_mask, overlapping2_mask = self._get_overlapping_images(
            mask_mode=True)
        cost_image[overlapping1_mask != 0] = 100
        cost_image[overlapping2_mask != 0] = 100

        cost_image[0: 2, :] = 255
        cost_image[-2:, :] = 255
        cost_image[:, 0:2] = 255
        cost_image[:, -2:] = 255
        return cost_image

    def _get_seamline(self):
        print(self.intersection_pts)
        times = self.downsample_times
        impro = ImageProcess(times)
        cost_im_downsample = impro.down_sample_2x2(self._get_cost_image())
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

    def _get_intersection_points_in_img(self):
        left = min([x[0] for x in self.intersection_pts])
        top = min([x[1] for x in self.intersection_pts])
        intersection_pts_in_im = [(x[0] - left, x[1] - top)
                                  for x in self.intersection_pts]
        return intersection_pts_in_im


class ImageTrans(object):

    def __init__(self, img1, img2, img1_mask, img2_mask, min_match_count=5):
        self.img1 = img1
        self.img2 = img2
        self.img1_mask = img1_mask
        self.img2_mask = img2_mask
        self.MIN_MATCH_COUNT = min_match_count

        # self.M1 = self._get_homography(method='surf')
        # self.M1 = self._get_homography()
        self.M1 = self._orb()
        self.M2, self.img1_corner_points_trans, self.img2_corner_points_trans = self._transform_corner_points()
        self.img1_trans, self.img2_trans = self._image_transform(
            mask_mode=False)
        self.img1_mask_trans, self.img2_mask_trans = self._image_transform(
            mask_mode=True)

    def __call__(self):
        return self.img1_corner_points_trans, self.img2_corner_points_trans, \
               self.img1_trans, self.img2_trans, \
               self.img1_mask_trans, self.img2_mask_trans, \
               self.M1, self.M2

    def _orb(self):
        # get m and matches points with sift algorithm
        tm = time.time()
        meth = cv.ORB_create()
        kp1, des1 = meth.detectAndCompute(self.img1, None)
        kp2, des2 = meth.detectAndCompute(self.img2, None)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for i in range(len(matches) - 1, -1, -1):
            if len(matches[i]) != 2:
                matches.pop(i)
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # M1, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            M1, _ = cv.findHomography(src_pts, dst_pts, cv.RHO, 1.0)
        print('orb algorithm consumed:', round(time.time() - tm, 1), 's')
        return M1

    def _get_homography(self, method='sift'):
        # get m and matches points with sift algorithm
        tm = time.time()
        assert method in ['surf', 'sift'], "invalid method to find homography"
        meth = cv.xfeatures2d.SURF_create(
            400) if method == 'surf' else cv.xfeatures2d.SIFT_create()
        kp1, des1 = meth.detectAndCompute(self.img1, None)
        kp2, des2 = meth.detectAndCompute(self.img2, None)

        FLANN_INDEX_KDTREE = 1
        index_prarams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_prarams, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # M1, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            M1, _ = cv.findHomography(src_pts, dst_pts, cv.RHO, 1.0)
        print('sift algorithm consumed:', round(time.time() - tm, 1), 's')
        return M1

    def _transform_corner_points(self):
        h1, w1 = self.img1.shape
        h2, w2 = self.img2.shape
        img1_corner_points = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1],
                                         [w1 - 1, 0]]).reshape(-1, 1, 2)

        left_sum = 0.
        top_sum = 0.
        for i in range(2):
            img1_corner_points_transformed = cv.perspectiveTransform(
                img1_corner_points, self.M1).squeeze()
            left, top = np.min(img1_corner_points_transformed, axis=0)
            left = left if left < 0 else 0
            top = top if top < 0 else 0
            self.M1[0, 2] += -left
            self.M1[1, 2] += -top
            left_sum += left
            top_sum += top
        M2 = np.float32([[1, 0, -left_sum], [0, 1, -top_sum]])
        img1_corner_points_transformed = cv.perspectiveTransform(
            img1_corner_points, self.M1).squeeze()
        img2_corner_points_transformed = np.float32([[-left_sum, -top_sum], [-left_sum, h2 - 1 - top_sum],
                                                     [w2 - 1 - left_sum, h2 - 1 - top_sum],
                                                     [w2 - 1 - left_sum, -top_sum]])
        return M2, img1_corner_points_transformed, img2_corner_points_transformed

    def _image_transform(self, mask_mode=False):
        trans_coordinates = np.vstack(
            (self.img1_corner_points_trans,
             self.img2_corner_points_trans))
        wmin, hmin = np.min(trans_coordinates, axis=0)
        wmax, hmax = np.max(trans_coordinates, axis=0)
        if mask_mode:
            img1_trans = cv.warpPerspective(
                self.img1_mask, self.M1, (wmax - wmin, hmax - hmin))
            img2_trans = cv.warpAffine(
                self.img2_mask, self.M2, (wmax - wmin, hmax - hmin))
        else:
            img1_trans = cv.warpPerspective(
                self.img1, self.M1, (wmax - wmin, hmax - hmin))
            img2_trans = cv.warpAffine(
                self.img2, self.M2, (wmax - wmin, hmax - hmin))
        return img1_trans, img2_trans


class ImageProcess(object):

    def __init__(self, times=1):
        self.times = times

    def down_sample_2x2(self, img):
        assert isinstance(self.times, int), "'times' must be an integer"
        image_downsample = [img]
        for _ in range(self.times):
            image_downsample.append(block_reduce(
                image_downsample[-1], (2, 2), func=np.max))

        return image_downsample

    def path_upsample(self, path, img):
        h, w = img.shape
        path_upsample = []
        for (x, y) in path:
            path_tmp = [(2 * x, 2 * y), (2 * x + 1, 2 * y),
                        (2 * x, 2 * y + 1), (2 * x + 1, 2 * y + 1)]
            path_upsample += [x for x in path_tmp if x[0] < w and x[1] < h]

        mask = np.zeros((h, w), dtype=np.float32)
        for s in path_upsample:
            mask[s[1], s[0]] = 0.01
        cost_modified = cv.multiply(mask, img)
        cost_modified[cost_modified == 0] = 255
        return cost_modified

    def intersection_points_in_im(self, t):
        pts_in_downsample = [t]
        for _ in range(self.times):
            x, y = pts_in_downsample[-1]
            x_downsample = max((x - 1) // 2 if x & 1 else x // 2, 0)
            y_downsample = max((y - 1) // 2 if y & 1 else y // 2, 0)
            pts_in_downsample.append((x_downsample, y_downsample))

        return pts_in_downsample

    def show_path(self):
        pass

    def get_most_points(self, intersection_points, img_pts_this, img_pts_another):
        ip = np.array(intersection_points, dtype=np.float32)
        # Ax +By + c = 0
        # A = y2 - y1
        # B = x1 - x2
        # C = x2 * y1 - x1 * y2
        B, a = ip[0] - ip[1]
        A = -a
        C = ip[1][0] * ip[0][1] - ip[0][0] * ip[1][1]
        tmp1, tmp2 = [], []
        for pts in img_pts_this:
            y = -A / B * pts[0] - C / B
            tmp1.append(pts) if y > pts[1] else tmp2.append(pts)
        # assert len(tmp1) == 3 or len(tmp2) == 3, 'points have been divided into 2 and 2'
        if len(tmp1) == 3 or len(tmp2) == 3:
            tmp = tmp1 if len(tmp1) == 3 else tmp2
            res = [(0., 0.)] * 3
            min_interpts0 = 2 ** 32 - 1
            min_interpts1 = 2 ** 32 - 1
            for pts in tmp:
                dis0 = math.hypot((ip[0] - pts)[0], (ip[0] - pts)[1])
                dis1 = math.hypot((ip[1] - pts)[0], (ip[1] - pts)[1])
                if dis0 < min_interpts0:
                    min_interpts0 = dis0
                    res[0] = pts
                if dis1 < min_interpts1:
                    min_interpts1 = dis1
                    res[-1] = pts
            for i, tp in enumerate(res):
                res[i] = tuple(tp)
            for i, tp in enumerate(tmp):
                tmp[i] = tuple(tp)
            for pts in tmp:
                res[1] = tuple(pts) if pts not in res else res[1]
        elif len(tmp1) == 2 and len(tmp2) == 2:
            img_pts_another = np.array(img_pts_another, dtype=np.float32)
            left_another, top_another = np.min(img_pts_another, axis=0)
            right_another, bottom_another = np.max(img_pts_another, axis=0)
            tmp = tmp2 if self.is_in_img(tmp1, left_another, right_another, top_another, bottom_another) else tmp1
            res = [(0., 0.)] * 2
            min_interpts0 = 2 ** 32 - 1
            min_interpts1 = 2 ** 32 - 1
            for pts in tmp:
                dis0 = math.hypot((ip[0] - pts)[0], (ip[0] - pts)[1])
                dis1 = math.hypot((ip[1] - pts)[0], (ip[1] - pts)[1])
                if dis0 < min_interpts0:
                    min_interpts0 = dis0
                    res[0] = pts
                if dis1 < min_interpts1:
                    min_interpts1 = dis1
                    res[-1] = pts
            for i, tp in enumerate(res):
                res[i] = tuple(tp)
        elif len(tmp1) == 4 or len(tmp2) == 4:
            tmp = tmp1 if len(tmp1) == 4 else tmp2
            res = [(0., 0.)] * 4
            min_interpts0 = 2 ** 32 - 1
            min_interpts1 = 2 ** 32 - 1
            for pts in tmp:
                dis0 = math.hypot((ip[0] - pts)[0], (ip[0] - pts)[1])
                dis1 = math.hypot((ip[1] - pts)[0], (ip[1] - pts)[1])
                if dis0 < min_interpts0:
                    min_interpts0 = dis0
                    res[0] = pts
                if dis1 < min_interpts1:
                    min_interpts1 = dis1
                    res[-1] = pts
            tmp = [x for x in tmp if tuple(x) != tuple(res[0]) and tuple(x) != tuple(res[-1])]
            min_interpts0 = 2 ** 32 - 1
            min_interpts1 = 2 ** 32 - 1
            for pts in tmp:
                dis0 = math.hypot((res[0] - pts)[0], (res[0] - pts)[1])
                dis1 = math.hypot((res[-1] - pts)[0], (res[-1] - pts)[1])
                if dis0 < min_interpts0:
                    min_interpts0 = dis0
                    res[1] = pts
                if dis1 < min_interpts1:
                    min_interpts1 = dis1
                    res[-2] = pts
            for i, tp in enumerate(res):
                res[i] = tuple(tp)
        return res

    def is_in_img(self, tmp, left_another, right_another, top_another, bottom_another):
        left = tmp[0][0] if tmp[0][0] < tmp[1][0] else tmp[1][0]
        right = tmp[0][0] if tmp[0][0] > tmp[1][0] else tmp[1][0]
        top = tmp[0][1] if tmp[0][1] < tmp[1][1] else tmp[1][1]
        bottom = tmp[0][1] if tmp[0][1] > tmp[1][1] else tmp[1][1]

        res = True if left_another < left and right_another > right \
                      and top_another < top and bottom_another > bottom else False
        return res
