import os
import sys
import random
import cv2
import numpy
import math
import albumentations as A
import traceback
import subprocess
import threading

from copy import deepcopy

sys.path.append('../ftools')
from ffile import fFile, fJson, fTxt, fXml
from fmark import fXmlMark, fJsonMark
from fimg import fImg
from fmath import fMath
from ftime import fTime

img_suffixes = ['.jpg', '.png', '.jpeg', '.bmp']


def detect_aug(self, config):
    """
    目标检测增强
    """
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    self.progress_bar_max = len(img_paths) * config['times']

    for img_path in img_paths:
        img_suffix = os.path.splitext(img_path)[1]
        img = fImg().read(img_path)
        h, w = img.shape[:2]
        xml_path = f'{os.path.splitext(img_path)[0]}.xml'
        _, img_size, xml_data = fXmlMark().read(xml_path)
        classes = [obj[0] for obj in xml_data]
        bboxes = [sum(obj[1], []) for obj in xml_data]
        for i in range(config['times']):
            allow_error_times = 10
            while allow_error_times > 0:
                try:
                    r = random.uniform(config['resize_low'], config['resize_high'])
                    t_h, t_w = round(h * r), round(w * r)
                    aug = A.Compose([
                        A.Resize(height=t_h, width=t_w, p=1),
                        A.PadIfNeeded(min_height=h, min_width=w, p=1, border_mode=0, value=config['mask_color'],
                                      mask_value=config['mask_color']),
                        A.HorizontalFlip(p=config['flip_x']),
                        A.VerticalFlip(p=config['flip_y']),
                        A.HueSaturationValue(
                            p=config['hsv_ratio'],
                            hue_shift_limit=config['hsv_limit'],
                            sat_shift_limit=config['hsv_limit'],
                            val_shift_limit=config['hsv_limit']
                        ),
                        A.Sharpen(p=config['sharpen']),
                        A.GaussNoise(p=config['gaussnoise']),
                        A.GaussianBlur(p=config['gaussblur']),
                        A.RandomBrightnessContrast(p=config['randombrightnesscontrast'],
                                                   brightness_limit=(
                                                       config['brightness_low'], config['brightness_high']),
                                                   contrast_limit=(config['contrast_low'], config['contrast_high'])),
                        A.Affine(p=config['affine'], cval=config['mask_color'], mode=cv2.BORDER_CONSTANT, cval_mask=0),
                        A.Perspective(keep_size=False, p=config['perspective'], pad_mode=cv2.BORDER_CONSTANT,
                                      mask_pad_val=config['mask_color'], pad_val=[0, 0, 0]),
                        A.ShiftScaleRotate(p=config['shift_scale_rotate'], border_mode=cv2.BORDER_CONSTANT,
                                           value=config['mask_color'], mask_value=config['mask_color'],
                                           rotate_limit=config['rotate_limit']),
                    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=50, min_visibility=0.5,
                                                label_fields=['class_labels']))
                    tr = aug(image=img, bboxes=bboxes, class_labels=classes)
                    tr_img = tr['image']
                    tr_bboxes = tr['bboxes']
                    tr_classes = tr['class_labels']
                    new_xml_data = [
                        [obj[0], [[obj[1][0], obj[1][1]], [obj[1][2], obj[1][3]]], 0]
                        for obj in list(zip(tr_classes, tr_bboxes))
                    ]
                    img_aug_path = img_path.replace(img_suffix, f'_aug{i}{img_suffix}').replace(config['scan_path'],
                                                                                                config['out_path'])
                    fImg().readin(img_aug_path, tr_img)
                    fXmlMark().readin(img_aug_path, (h, w), new_xml_data)
                    break
                except:
                    allow_error_times -= 1
            else:
                error(message=f'{fTime().format()}: {detect_aug.__name__}\n{img_path} out of aug times')

            self.progress_bar_value += 1


def detect_infer(self, config):
    """
    目标检测仿真
    """
    self.progress_bar_show = True
    config['running'] = True
    threading.Thread(target=detect_infer_cmd, args=(self, config)).start()
    while config['running']: self.update()


def detect_infer_cmd(self, config):
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    if len(img_paths) == 0:
        message, message_type = '图片为空', 'warning'
    else:
        if config['out_path'] == '': config['out_path'] = config['scan_path']
        os.makedirs(config['out_path'], exist_ok=True)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        process = subprocess.Popen(['cmd'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, universal_newlines=True, startupinfo=startupinfo)
        commands = [
            config['exe_path'][:2],
            f'cd {os.path.dirname(config["exe_path"])}',
            f'{os.path.basename(config["exe_path"])} -a {config["model_path"]} {config["ini_path"]} {config["scan_path"]} {config["out_path"]}'
        ]
        for command in commands: process.stdin.write(f'{command}\n')
        process.stdin.flush()
        output, errors = process.communicate()
        if output.count('XML file saved successfully.') == len(img_paths) or output.count('get detect num') == len(img_paths):
            message, message_type = '仿真成功', 'info'
        else:
            error(path=f'./error/{fTime().format()}.txt', message=f'-----\nerrors: {errors}\n-----\noutput: {output}')
            message, message_type = '仿真失败', 'error'
    config['running'] = False
    self.progress_bar_show = False
    self.message(message, message_type)


def detect_convert(self, config):
    """
    目标检测模型转换
    """
    self.progress_bar_show = True
    threading.Thread(target=detect_convert_cmd, args=(self, config)).start()
    config['running'] = True
    while config['running']: self.update()


def detect_convert_cmd(self, config):
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    process = subprocess.Popen(['cmd'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, universal_newlines=True, startupinfo=startupinfo)
    model_out_path = f'{config["model_out_folder"]}{config["model_out_name"]}'
    commands = [
        config['exe_path'][:2],
        f'cd {os.path.dirname(config["exe_path"])}',
        f'{os.path.basename(config["exe_path"])} -s {config["model_in_path"]} {model_out_path} {config["ini_path"]}'
    ]
    for command in commands: process.stdin.write(f'{command}\n')
    process.stdin.flush()
    output, errors = process.communicate()
    if output.find('convert scussce\ncreate restult 1') != -1:
        os.makedirs(config['model_out_folder'], exist_ok=True)
        os.rename(f'{os.path.split(config["model_in_path"])[0]}/model.encode', model_out_path)
        message, message_type = '转换成功', 'info'
    else:
        error(path=f'./error/{fTime().format()}.txt', message=f'-----\nerrors: {errors}\n-----\noutput: {output}')
        message, message_type = '转换失败', 'error'
    config['running'] = False
    self.progress_bar_show = False
    self.message(message, message_type)


def segment_aug(self, config):
    """
    分割增强
    """
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    self.progress_bar_max = len(img_paths) * config['times']

    for img_path in img_paths:
        img_suffix = os.path.splitext(img_path)[1]
        img = fImg().read(img_path)
        h, w = img.shape[:2]
        img_mask = numpy.ones_like(img, dtype=numpy.uint8) * 0  # 生成同size的黑图
        json_path = f'{os.path.splitext(img_path)[0]}.json'
        _, img_size, json_data = fJsonMark().read(json_path)
        labels = list(set([shape[0] for shape in json_data]))
        marks = dict()
        for label in labels:
            points = [numpy.array(shape[1], dtype=numpy.int32) for shape in
                      list(filter(lambda shape: shape[0] == label, json_data))]
            marks[label] = deepcopy(img_mask)
            cv2.fillPoly(marks[label], points, color=[255, 255, 255])
        labels, masks = list(marks.keys()), list(marks.values())
        for i in range(config['times']):
            t_h, t_w = round(h * random.uniform(config['resize_h_low'], config['resize_h_high'])), round(
                w * random.uniform(config['resize_w_low'], config['resize_w_high']))
            aug = A.Compose([
                A.Resize(height=t_h, width=t_w, p=1),
                A.PadIfNeeded(min_height=h, min_width=w, p=1, mask_value=config['mask_color'],
                              border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=config['flip_x']),
                A.VerticalFlip(p=config['flip_y']),
                A.HueSaturationValue(
                    p=config['hsv_ratio'],
                    hue_shift_limit=config['hsv_limit'],
                    sat_shift_limit=config['hsv_limit'],
                    val_shift_limit=config['hsv_limit']
                ),
                A.Sharpen(p=config['sharpen']),
                A.GaussNoise(p=config['gaussnoise']),
                A.GaussianBlur(p=config['gaussblur']),
                A.RandomBrightnessContrast(p=config['randombrightnesscontrast'],
                                           brightness_limit=(config['brightness_low'], config['brightness_high']),
                                           contrast_limit=(config['contrast_low'], config['contrast_high'])),
                A.ElasticTransform(p=config['elastic_transform'], border_mode=cv2.BORDER_CONSTANT,
                                   value=config['mask_color']),
                A.GridDistortion(p=config['grid_distortion'], border_mode=cv2.BORDER_CONSTANT,
                                 value=config['mask_color']),
                A.OpticalDistortion(p=config['optical_distortion'], border_mode=cv2.BORDER_CONSTANT,
                                    value=config['mask_color']),
                A.ShiftScaleRotate(p=config['shift_scale_rotate'], border_mode=cv2.BORDER_CONSTANT,
                                   value=config['mask_color'], mask_value=config['mask_color'],
                                   rotate_limit=config['rotate_limit']),
            ])
            augmented = aug(image=img, masks=masks)
            img_aug = augmented['image']
            img_masks = augmented['masks']
            img_aug_path = img_path.replace(img_suffix, f'_aug{i}{img_suffix}').replace(config['scan_path'],
                                                                                        config['out_path'])
            new_json_data = list()

            for i, img_mask in enumerate(img_masks):
                img0 = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
                th, img0 = cv2.threshold(img0, 200, 255, cv2.THRESH_BINARY)
                contours, hi = cv2.findContours(img0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = [[contour, cv2.contourArea(contour)] for contour in contours]
                contours.sort(key=lambda _: _[1], reverse=True)
                for contour in contours:
                    new_json_data.append(
                        [labels[i], [[int(point[0][0]), int(point[0][1])] for point in contour[0]], 'polygon'])
            fImg().readin(img_aug_path, img_aug)
            fJsonMark().readin(img_aug_path, img_aug.shape[:2], new_json_data)

            self.progress_bar_value += 1


def obb_aug(self, config):
    """
    obb增强
    """
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    self.progress_bar_max = len(img_paths) * config['times']

    for img_path in img_paths:
        img_suffix = os.path.splitext(img_path)[1]
        img = fImg().read(img_path)
        h, w = img.shape[:2]
        json_path = f'{os.path.splitext(img_path)[0]}.json'
        _, img_size, json_data = fJsonMark().read(json_path)
        rotate_x, rotate_y = w / 2, h / 2
        for i in range(config['times']):
            allow_error_times = 10
            while allow_error_times > 0:
                try:
                    angle = random.randrange(config['angle_low'], config['angle_high'], config['angle_range'])
                    angle_pi = -angle * math.pi / 180
                    aug_img = cv2.warpAffine(img, cv2.getRotationMatrix2D((rotate_x, rotate_y), angle, 1), (w, h),
                                             borderValue=config['mask_color'])
                    aug_json_data = deepcopy(json_data)
                    for shape in aug_json_data:
                        for point in shape[1]:
                            x, y = point
                            point[0] = (x - rotate_x) * math.cos(angle_pi) - (y - rotate_y) * math.sin(
                                angle_pi) + rotate_x
                            point[1] = (x - rotate_x) * math.sin(angle_pi) + (y - rotate_y) * math.cos(
                                angle_pi) + rotate_y
                            assert 0 <= point[0] <= w and 0 <= point[1] <= h

                    aug_img_h, aug_img_w = aug_img.shape[:2]
                    flip_x_random, flip_y_random = random.randint(0, 1), random.randint(0, 1)
                    if config['flip_x'] and flip_x_random: aug_img = cv2.flip(aug_img, 1)
                    if config['flip_y'] and flip_y_random: aug_img = cv2.flip(aug_img, 0)
                    for shape in aug_json_data:
                        for point in shape[1]:
                            if config['flip_x'] and flip_x_random: point[0] = aug_img_w - point[0]
                            if config['flip_y'] and flip_y_random: point[1] = aug_img_h - point[1]

                    aug = A.Compose([
                        A.HueSaturationValue(
                            p=config['hsv_ratio'],
                            hue_shift_limit=config['hsv_limit'],
                            sat_shift_limit=config['hsv_limit'],
                            val_shift_limit=config['hsv_limit']
                        ),
                        A.Sharpen(p=config['sharpen']),
                        A.GaussNoise(p=config['gaussnoise']),
                        A.GaussianBlur(p=config['gaussblur']),
                        A.RandomBrightnessContrast(p=config['randombrightnesscontrast'],
                                                   brightness_limit=(
                                                       config['brightness_low'], config['brightness_high']),
                                                   contrast_limit=(config['contrast_low'], config['contrast_high'])),
                    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=50, min_visibility=0.5,
                                                label_fields=['class_labels']))
                    tr = aug(image=aug_img, bboxes=[], class_labels=[])
                    aug_img = tr['image']

                    aug_img_path = img_path.replace(img_suffix, f'_aug{i}{img_suffix}').replace(config['scan_path'],
                                                                                                config['out_path'])
                    fImg().readin(aug_img_path, aug_img)
                    fJsonMark().readin(aug_img_path, aug_img.shape[:2], aug_json_data)
                    break
                except Exception:
                    allow_error_times -= 1
            else:
                error(message=f'{fTime().format()}: {obb_aug.__name__}\n{img_path} out of aug times')

            self.progress_bar_value += 1


def obb_crop(self, config):
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    self.progress_bar_max = len(img_paths) * 3

    for img_path in img_paths:
        img_suffix = os.path.splitext(img_path)[1]
        img = fImg().read(img_path)
        img_h, img_w = img.shape[:2]
        json_path = f'{os.path.splitext(img_path)[0]}.json'
        _, img_size, json_data = fJsonMark().read(json_path)
        crop_img_h = img_h // 3
        for i in range(3):
            ymin, ymax = i * crop_img_h, (i + 1) * crop_img_h
            crop_json_data = list(filter(lambda shape: ymin < (shape[1][0][1] + shape[1][1][1]) / 2 < ymax, json_data))
            y = [point[1] for point in sum([shape[1] for shape in json_data], [])]
            if len(y) > 1: ymin, ymax = min(ymin, min(y)), max(ymax, max(y))
            crop_img = img[ymin:ymax, :]
            for shape in crop_json_data:
                for point in shape[1]:
                    point[1] -= ymin
            crop_img_path = img_path.replace(img_suffix, f'_crop{i}{img_suffix}').replace(config['scan_path'],
                                                                                          config['out_path'])
            fImg().readin(crop_img_path, crop_img)
            fJsonMark().readin(crop_img_path, crop_img.shape[:2], crop_json_data)

            self.progress_bar_value += 1


def obb_splice(self, config):
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    self.progress_bar_max = len(img_paths)

    numpy.random.shuffle(img_paths)
    if len(img_paths) % 3: img_paths.extend(img_paths[:3 - len(img_paths) % 3])
    for i in range(len(img_paths) // 3):
        img1_path, img2_path, img3_path = img_paths[i * 3: (i + 1) * 3]
        img1, img2, img3 = fImg().read(img1_path), fImg().read(img2_path), fImg().read(img3_path)
        img1_h, img1_w = img1.shape[:2]
        img2_h, img2_w = img2.shape[:2]
        img3_h, img3_w = img3.shape[:2]
        json1_data = fJsonMark().read(f'{os.path.splitext(img1_path)[0]}.json')[-1]
        json2_data = fJsonMark().read(f'{os.path.splitext(img2_path)[0]}.json')[-1]
        json3_data = fJsonMark().read(f'{os.path.splitext(img3_path)[0]}.json')[-1]
        img_w = max(img1_w, img2_w, img3_w)
        if img_w > img1_w: img1 = cv2.copyMakeBorder(img1, 0, 0, 0, img_w - img1_w, cv2.BORDER_REPLICATE)
        if img_w > img2_w: img2 = cv2.copyMakeBorder(img2, 0, 0, 0, img_w - img2_w, cv2.BORDER_REPLICATE)
        if img_w > img3_w: img3 = cv2.copyMakeBorder(img3, 0, 0, 0, img_w - img3_w, cv2.BORDER_REPLICATE)
        img = numpy.vstack((img1, img2, img3))
        for shape in json2_data:
            for point in shape[1]:
                point[1] += img1_h
        for shape in json3_data:
            for point in shape[1]:
                point[1] += (img1_h + img2_h)
        json1_data.extend(json2_data)
        json1_data.extend(json3_data)
        img_path = f'{config["out_path"]}/{config["prefix"]}_{str(i):0>4}_{random.randint(0, 99999999):0>8}.png'
        fImg().readin(img_path, img)
        fJsonMark().readin(img_path, img.shape[:2], json1_data)

        self.progress_bar_value += 3


def pt_cv_quan(self, config):
    """
    胶粒画外圈
    """
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    self.progress_bar_max = len(img_paths)

    for img_path in img_paths:
        img = fImg().read(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        xml_path = f'{os.path.splitext(img_path)[0]}.xml'
        _, img_size, xml_data = fXmlMark().read(xml_path)
        th, img0 = cv2.threshold(img, config['color_low'], config['color_high'], cv2.THRESH_BINARY)
        img0 = cv2.bitwise_not(img0)
        img_close = cv2.morphologyEx(img0, cv2.MORPH_CLOSE, numpy.ones((config['box'], config['box']), numpy.uint8), 3)
        img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, numpy.ones((config['box'], config['box']), numpy.uint8),
                                    3)
        contours, hi = cv2.findContours(img_open, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for contour in contours:
            x = [point[0][0] for point in contour]
            y = [point[0][1] for point in contour]
            rectangles.append([min(x), min(y), max(x), max(y)])
        rectangles.sort(key=lambda shape: (shape[2] - shape[0]) * (shape[3] - shape[1]), reverse=True)
        x1, y1, x2, y2 = rectangles[0]
        if config['label_type']:
            xml_data = list(filter(lambda shape: shape[0] != config['label'], xml_data))
        xml_data.insert(0, [config['label'], [[x1, y1], [x2, y2]], 0])
        fXmlMark().readin(img_path, img_size, xml_data)

        self.progress_bar_value += 1


def segment_optimize(self, config):
    """
    优化分割模型的infer结果
    """
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    self.progress_bar_max = len(img_paths)

    for img_path in img_paths:
        json_path = f'{os.path.splitext(img_path)[0]}.json'
        _, img_size, json_data = fJsonMark().read(json_path)
        for shape in json_data:
            # 点距
            new_points = list()
            for point in shape[1]:
                if len(new_points) == 0 or fMath().distance(point, new_points[-1]) > config['point_distance']:
                    new_points.append(point)
            shape[1] = deepcopy(new_points)
        fJsonMark().readin(img_path.replace(config['scan_path'], config['out_path']), img_size, json_data)

        self.progress_bar_value += 1


def segment_connect_hollow(self, config):
    """
    连通jg和in
    """
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    self.progress_bar_max = len(img_paths)

    for img_path in img_paths:
        json_path = f'{os.path.splitext(img_path)[0]}.json'
        _, img_size, json_data = fJsonMark().read(json_path)
        # 获取第一个jg和第一个in的点集
        try:
            jg = list(filter(lambda shape: shape[0] == 'jg', json_data))[0][1]
            try:
                in0 = list(filter(lambda shape: shape[0] == 'in', json_data))[0][1]
            except IndexError:
                in0 = list(filter(lambda shape: shape[0] == 'jg', json_data))[1][1]
            json_data = list(filter(lambda shape: shape[0] not in ['jg', 'in'], json_data))
        except IndexError:
            error(message=f'{fTime().format()}: {segment_connect_hollow.__name__}\n{traceback.format_exc()}')
            continue
        new_points = deepcopy(jg)
        distances = [fMath().distance(jg[0], point) for point in in0]
        min_distance_index = distances.index(min(distances))
        nearest_point = in0[min_distance_index]  # 最近连通点
        new_points.extend(in0[min_distance_index:])
        new_points.extend(in0[:min_distance_index])
        new_points.append(nearest_point)
        new_points.append(jg[-1])
        new_shape = ['jg', new_points, 'polygon']
        json_data.append(new_shape)
        fJsonMark().readin(img_path.replace(config['scan_path'], config['out_path']), img_size, json_data)

        self.progress_bar_value += 1


def random_move(self, config):
    """随机移出图片"""
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    numpy.random.shuffle(img_paths)
    img_paths = img_paths[:round(len(img_paths) * config['random_ratio'])]
    self.progress_bar_max = len(img_paths)

    for img_path in img_paths:
        new_img_path = img_path.replace(config['scan_path'], config['out_path'])
        fFile().copy(img_path, new_img_path, is_move=True)
        for mark_suffix in ['.xml', '.json']:
            mark_path = f'{os.path.splitext(img_path)[0]}{mark_suffix}'
            mark_tool = get_mark_tool(mark_suffix)
            _, img_size, mark_data = mark_tool.read(mark_path)
            if os.path.exists(mark_path): os.remove(mark_path)
            mark_tool.readin(new_img_path, img_size, mark_data)

        self.progress_bar_value += 1


def rename_img(self, config):
    """
    研瑞根据相机号重命名图片
    """
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    self.progress_bar_max = len(img_paths)

    for img_path in img_paths:
        img_path_name, img_suffix = os.path.splitext(img_path)
        new_img_path_name = img_path_name.split('_')
        if new_img_path_name[-1][0] == 'c':
            new_img_path_name[-1] = f'c{config["camera"]}'
        else:
            new_img_path_name.append(f'c{config["camera"]}')
        new_img_path_name = '_'.join(new_img_path_name)
        os.rename(img_path, img_path.replace(img_path_name, new_img_path_name))
        for mark_suffix in ['.xml', '.json']:
            mark_path = f'{os.path.splitext(img_path)[0]}{mark_suffix}'
            if os.path.exists(mark_path): os.rename(mark_path, mark_path.replace(img_path_name, new_img_path_name))

        self.progress_bar_value += 1


def filter_img(self, config):
    """
    根据code和conf筛选图片, 提供是否仅含选项
    """
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    self.progress_bar_max = len(img_paths)

    config['code'] = str(config['code'])
    for img_path in img_paths:
        for mark_suffix in ['.xml', '.json']:
            mark_path = f'{os.path.splitext(img_path)[0]}{mark_suffix}'
            mark_tool = get_mark_tool(mark_suffix)
            _, img_size, mark_data = mark_tool.read(mark_path)
            if len(mark_data) > 0:
                match_count = 0
                for shape in mark_data:
                    if config['code'] != '' and config['code'] != shape[0]: continue
                    if config['conf_low'] != '' and config['conf_low'] > shape[-1]: continue
                    if config['conf_high'] != '' and config['conf_high'] < shape[-1]: continue
                    match_count += 1
                if match_count > 0:
                    if not config['only_has'] or (config['only_has'] and match_count == len(mark_data)):
                        if os.path.exists(img_path): fFile().copy(img_path, img_path.replace(config['scan_path'],
                                                                                             config['out_path']),
                                                                  is_move=True)
                        fFile().copy(mark_path, mark_path.replace(config['scan_path'], config['out_path']),
                                     is_move=True)

        self.progress_bar_value += 1


def clean_conf(self, config):
    """
    清空置信度
    """
    img_paths = fFile().scan(config['scan_path'], img_suffixes)
    self.progress_bar_max = len(img_paths)
    for img_path in img_paths:
        for mark_suffix in ['.xml', '.json']:
            mark_path = f'{os.path.splitext(img_path)[0]}{mark_suffix}'
            mark_tool = get_mark_tool(mark_suffix)
            _, img_size, mark_data = mark_tool.read(mark_path)
            for shape in mark_data:
                shape[-1] = -1
            mark_tool.readin(img_path, img_size, mark_data)
        self.progress_bar_value += 1


def batch_modify_camera(self, config):
    """
    批量修改相机参数
    """
    modify_items = {
        "曝光时间": "ExposureTime",
        "红色增益": "RedGain",
        "绿色增益": "GreenGain",
        "蓝色增益": "BlueGain",
        "增益": "Gain",
        "图像宽度": "ImageWidth",
        "图像高度": "ImageHeight",
        "宽度偏移": "WidthOffset",
        "高度偏移": "HeightOffset",
    }
    assert config['modify_item'] in modify_items.keys(), f'error modify_item: {config["modify_item"]}'
    modify_item = modify_items[config['modify_item']]
    cameras = [config[f'camera{i}'] for i in range(1, 17)]
    product_config_path = f'{config["configs_path"]}/ProductConfig.xml'
    product_config = fXml().read(product_config_path)
    product_config = product_config['ArrayOfProductModel']['ProductModel']
    templates = [[template['Order'], template['Name']] for template in product_config]
    if len(config['config']) > 0: templates = list(filter(lambda template: template[1] == config['config'], templates))
    assert len(templates) > 0, 'len(templates) == 0'
    for template in templates:
        order, name = template
        camera_config_path = f'{config["configs_path"]}/{order}/CameraConfig.xml'
        camera_config = fXml().read(camera_config_path)
        for camera in camera_config['ArrayOfCameraConfigModel']['CameraConfigModel']:
            if cameras[int(camera['Order']) - 1] is False: continue
            if config['modify_type']:
                if int(camera[modify_item]) == camera[modify_item]:
                    camera[modify_item] = int(camera[modify_item])
                else:
                    camera[modify_item] = float(camera[modify_item])
                camera[modify_item] += config['modify_data']
            else:
                camera[modify_item] = config['modify_data']
        fXml().readin(camera_config_path, camera_config)


def get_mark_tool(mark_suffix):
    """
    返回标注文件工具包
    """
    try:
        mark_tool = [fXmlMark(), fJsonMark()][['.xml', '.json'].index(mark_suffix)]
    except IndexError:
        mark_tool = None
        error(message=f'{fTime().format()}: {get_mark_tool.__name__}\n{traceback.format_exc()}')
    return mark_tool


def error(path=f'./error/{fTime().format()}.txt', message='\n'):
    fTxt().add(path, message)


if __name__ == '__main__':
    pass
