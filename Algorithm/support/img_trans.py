from __future__ import print_function

import argparse
import time
from collections import OrderedDict

import cv2 as cv
import numpy as np

EXPOS_COMP_CHOICES = OrderedDict()
EXPOS_COMP_CHOICES['gain_blocks'] = cv.detail.ExposureCompensator_GAIN_BLOCKS
EXPOS_COMP_CHOICES['gain'] = cv.detail.ExposureCompensator_GAIN
EXPOS_COMP_CHOICES['channel'] = cv.detail.ExposureCompensator_CHANNELS
EXPOS_COMP_CHOICES['channel_blocks'] = cv.detail.ExposureCompensator_CHANNELS_BLOCKS
EXPOS_COMP_CHOICES['no'] = cv.detail.ExposureCompensator_NO

BA_COST_CHOICES = OrderedDict()
BA_COST_CHOICES['ray'] = cv.detail_BundleAdjusterRay
BA_COST_CHOICES['reproj'] = cv.detail_BundleAdjusterReproj
BA_COST_CHOICES['affine'] = cv.detail_BundleAdjusterAffinePartial
BA_COST_CHOICES['no'] = cv.detail_NoBundleAdjuster

FEATURES_FIND_CHOICES = OrderedDict()
try:
    FEATURES_FIND_CHOICES['surf'] = cv.xfeatures2d_SURF.create
except AttributeError:
    print("SURF not available")
# if SURF not available, ORB is default
FEATURES_FIND_CHOICES['orb'] = cv.ORB.create
try:
    FEATURES_FIND_CHOICES['sift'] = cv.SIFT.create
except AttributeError:
    print("SIFT not available")
try:
    FEATURES_FIND_CHOICES['brisk'] = cv.BRISK_create
except AttributeError:
    print("BRISK not available")
try:
    FEATURES_FIND_CHOICES['akaze'] = cv.AKAZE_create
except AttributeError:
    print("AKAZE not available")

ESTIMATOR_CHOICES = OrderedDict()
ESTIMATOR_CHOICES['homography'] = cv.detail_HomographyBasedEstimator
ESTIMATOR_CHOICES['affine'] = cv.detail_AffineBasedEstimator

WARP_CHOICES = (
    'spherical',
    'plane',
    'affine',
    'cylindrical',
    'fisheye',
    'stereographic',
    'compressedPlaneA2B1',
    'compressedPlaneA1.5B1',
    'compressedPlanePortraitA2B1',
    'compressedPlanePortraitA1.5B1',
    'paniniA2B1',
    'paniniA1.5B1',
    'paniniPortraitA2B1',
    'paniniPortraitA1.5B1',
    'mercator',
    'transverseMercator',
)

WAVE_CORRECT_CHOICES = ('horiz', 'no', 'vert',)

BLEND_CHOICES = ('multiband', 'feather', 'no',)


class ImageTrans(object):

    def __init__(self, img_name):
        self.img_name = img_name

    def __call__(self):
        self.main()

    def get_matcher(self):
        try_cuda = False
        matcher_type = 'affine'
        match_conf = 0.3
        # match_conf=0.65

        range_width = -1
        if matcher_type == "affine":
            matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
        elif range_width == -1:
            matcher = cv.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)
        else:
            matcher = cv.detail.BestOf2NearestRangeMatcher_create(range_width, try_cuda, match_conf)
        return matcher

    def get_compensator(self):
        expos_comp_type = EXPOS_COMP_CHOICES['channel_blocks']
        expos_comp_nr_feeds = 1
        expos_comp_block_size = 32
        # expos_comp_nr_filtering = args.expos_comp_nr_filtering
        if expos_comp_type == cv.detail.ExposureCompensator_CHANNELS:
            compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)
            # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
        elif expos_comp_type == cv.detail.ExposureCompensator_CHANNELS_BLOCKS:
            compensator = cv.detail_BlocksChannelsCompensator(
                expos_comp_block_size, expos_comp_block_size,
                expos_comp_nr_feeds
            )
            # compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
        else:
            compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
        return compensator

    def main(self):
        # mask_dir = 'mask2/'
        mask_dir = 'roadmask/'
        img_names = self.img_name

        print(img_names)
        work_megapix = 0.6
        seam_megapix = 0.1
        compose_megapix = -1
        conf_thresh = 1.0
        ba_refine_mask = 'xxxxx'

        wave_correct = 'no'
        if wave_correct == 'no':
            do_wave_correct = False
        else:
            do_wave_correct = True

        save_graph = True
        # if args.save_graph is None:
        #     save_graph = False
        # else:
        #     save_graph = True
        warp_type = 'affine'
        # blend_type = 'no'
        # blend_strength = 5
        # result_name = 'result.JPG'
        timelapse = True
        timelapse_type = cv.detail.Timelapser_AS_IS
        # if args.timelapse is not None:
        #     timelapse = True
        #     if args.timelapse == "as_is":
        #         timelapse_type = cv.detail.Timelapser_AS_IS
        #     elif args.timelapse == "crop":
        #         timelapse_type = cv.detail.Timelapser_CROP
        #     else:
        #         print("Bad timelapse method")
        #         exit()
        # else:
        #     timelapse = False
        finder = FEATURES_FIND_CHOICES['sift']()
        seam_work_aspect = 1
        full_img_sizes = []
        features = []
        images = []
        is_work_scale_set = False
        is_seam_scale_set = False
        is_compose_scale_set = False
        for name in img_names:
            full_img = cv.imread(name)
            if full_img is None:
                print("Cannot read image ", name)
                exit()
            full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
            if work_megapix < 0:
                img = full_img
                work_scale = 1
                is_work_scale_set = True
            else:
                if is_work_scale_set is False:
                    work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                    is_work_scale_set = True
                img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale,
                                interpolation=cv.INTER_LINEAR_EXACT)
            if is_seam_scale_set is False:
                seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                seam_work_aspect = seam_scale / work_scale
                is_seam_scale_set = True
            img_feat = cv.detail.computeImageFeatures2(finder, img)
            features.append(img_feat)
            img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
            images.append(img)

        matcher = self.get_matcher()
        p = matcher.apply2(features)
        matcher.collectGarbage()

        if save_graph:
            with open('save_graph.txt', 'w') as fh:
                fh.write(cv.detail.matchesGraphAsString(img_names, p, conf_thresh))

        indices = cv.detail.leaveBiggestComponent(features, p, 0.3)
        img_subset = []
        img_names_subset = []
        full_img_sizes_subset = []
        for i in range(len(indices)):
            img_names_subset.append(img_names[indices[i, 0]])
            img_subset.append(images[indices[i, 0]])
            full_img_sizes_subset.append(full_img_sizes[indices[i, 0]])
        images = img_subset
        img_names = img_names_subset
        full_img_sizes = full_img_sizes_subset
        num_images = len(img_names)
        if num_images < 2:
            print("Need more images")
            exit()

        estimator = ESTIMATOR_CHOICES['affine']()
        b, cameras = estimator.apply(features, p, None)
        if not b:
            print("affine estimation failed.")
            exit()
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)

        adjuster = BA_COST_CHOICES['affine']()
        adjuster.setConfThresh(1)
        refine_mask = np.zeros((3, 3), np.uint8)
        if ba_refine_mask[0] == 'x':
            refine_mask[0, 0] = 1
        if ba_refine_mask[1] == 'x':
            refine_mask[0, 1] = 1
        if ba_refine_mask[2] == 'x':
            refine_mask[0, 2] = 1
        if ba_refine_mask[3] == 'x':
            refine_mask[1, 1] = 1
        if ba_refine_mask[4] == 'x':
            refine_mask[1, 2] = 1
        adjuster.setRefinementMask(refine_mask)
        b, cameras = adjuster.apply(features, p, cameras)
        if not b:
            print("Camera parameters adjusting failed.")
            exit()
        focals = []
        for cam in cameras:
            focals.append(cam.focal)
        focals.sort()
        if len(focals) % 2 == 1:
            warped_image_scale = focals[len(focals) // 2]
        else:
            warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
        if do_wave_correct:
            rmats = []
            for cam in cameras:
                rmats.append(np.copy(cam.R))
            rmats = cv.detail.waveCorrect(rmats, cv.detail.WAVE_CORRECT_HORIZ)
            for idx, cam in enumerate(cameras):
                cam.R = rmats[idx]
        corners = []
        masks_warped = []
        images_warped = []
        sizes = []
        masks = []
        for i in range(0, num_images):
            um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
            masks.append(um)

        warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
        for idx in range(0, num_images):
            K = cameras[idx].K().astype(np.float32)
            swa = seam_work_aspect
            K[0, 0] *= swa
            K[0, 2] *= swa
            K[1, 1] *= swa
            K[1, 2] *= swa
            corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            corners.append(corner)
            sizes.append((image_wp.shape[1], image_wp.shape[0]))
            images_warped.append(image_wp)
            p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            masks_warped.append(mask_wp.get())

        # images_warped_f = []
        # for img in images_warped:
        #     imgf = img.astype(np.float32)
        #     images_warped_f.append(imgf)

        compensator = self.get_compensator()
        compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

        # seam_finder = SEAM_FIND_CHOICES['gc_color']
        # seam_finder.find(images_warped_f, corners, masks_warped)
        compose_scale = 1
        corners = []
        sizes = []
        blender = None
        timelapser = None
        timelapser_mask = None
        # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
        for idx, name in enumerate(img_names):
            print(f'{idx + 1}/{len(img_names)}', sep=',', end=',')
            full_img = cv.imread(name)
            full_img_mask = cv.imread(mask_dir + 'mask_' + name.split('/')[-1][:-4]+'.png')
            if not is_compose_scale_set:
                if compose_megapix > 0:
                    compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_compose_scale_set = True
                compose_work_aspect = compose_scale / work_scale
                warped_image_scale *= compose_work_aspect
                warper = cv.PyRotationWarper(warp_type, warped_image_scale)
                for i in range(0, len(img_names)):
                    cameras[i].focal *= compose_work_aspect
                    cameras[i].ppx *= compose_work_aspect
                    cameras[i].ppy *= compose_work_aspect
                    sz = (full_img_sizes[i][0] * compose_scale, full_img_sizes[i][1] * compose_scale)
                    K = cameras[i].K().astype(np.float32)
                    roi = warper.warpRoi(sz, K, cameras[i].R)
                    corners.append(roi[0:2])
                    sizes.append(roi[2:4])
            if abs(compose_scale - 1) > 1e-1:
                img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                                interpolation=cv.INTER_LINEAR_EXACT)
            else:
                img = full_img
            _img_size = (img.shape[1], img.shape[0])
            K = cameras[idx].K().astype(np.float32)
            corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
            _, image_mask_warped = warper.warp(full_img_mask, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            compensator.apply(idx, corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            image_mask_warped_s = image_mask_warped.astype(np.int16)

            # dilated_mask = cv.dilate(masks_warped[idx], None)
            # seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            # mask_warped = cv.bitwise_and(seam_mask, mask_warped)
            if blender is None and not timelapse:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
                blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
                if blend_width < 1:
                    blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                elif blend_type == "multiband":
                    blender = cv.detail_MultiBandBlender()
                    blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
                elif blend_type == "feather":
                    blender = cv.detail_FeatherBlender()
                    blender.setSharpness(1. / blend_width)
                blender.prepare(dst_sz)
            elif timelapser is None and timelapse:
                timelapser = cv.detail.Timelapser_createDefault(timelapse_type)
                timelapser.initialize(corners, sizes)
                timelapser_mask = cv.detail.Timelapser_createDefault(timelapse_type)
                timelapser_mask.initialize(corners, sizes)
            if timelapse:
                # a = []
                ma_tones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
                timelapser.process(image_warped_s, ma_tones, corners[idx])
                timelapser_mask.process(image_mask_warped_s, ma_tones, corners[idx])
                pos_s = img_names[idx].rfind("/")
                if pos_s == -1:
                    fixed_file_name = "fixed_" + img_names[idx]
                    fixed_mask_file_name = mask_dir + "fixed_mask_" + img_names[idx]
                else:
                    fixed_file_name = img_names[idx][:pos_s + 1] + "fixed_" + img_names[idx][pos_s + 1:]
                    fixed_mask_file_name = mask_dir + "fixed_mask_" + img_names[idx][pos_s + 1:]

                cv.imwrite(fixed_file_name, timelapser.getDst())
                # a.append(np.array(cv.imread(fixed_file_name, 0)))
                cv.imwrite(fixed_mask_file_name, timelapser_mask.getDst())

        #     else:
        #         blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
        # if not timelapse:
        #     result = None
        #     result_mask = None
        #     result, result_mask = blender.blend(result, result_mask)
        #     cv.imwrite(result_name, result)
        #     zoom_x = 600.0 / result.shape[1]
        #     dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        #     dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
        #     cv.imshow(result_name, dst)
        #     cv.waitKey()

        # corners = np.array(corners)
        # sizes = np.array(sizes)
        # left_top = corners
        # right_bottom = corners + sizes
        #
        # x_min, y_min = np.min(left_top, axis=0)
        # x_min = x_min if x_min < 0 else 0
        # y_min = y_min if y_min < 0 else 0
        # left_top += [-x_min, -y_min]
        # right_bottom += [-x_min, -y_min]
        # res = dict()
        # for i in range(len(img_names)):
        #     res[(tuple(left_top[i]), tuple(right_bottom[i]))] = (img_names[i], left_top[i], right_bottom[i])


if __name__ == '__main__':
    img_names = []
    for i in range(4):
        img_names.append('DSC0072{}.JPG'.format(i))
    tm = time.time()
    res = ImageTrans(img_names)()
    print(round(time.time() - tm, 2), 's', sep='')
