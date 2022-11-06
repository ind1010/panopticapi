#!/usr/bin/env python
'''
This script converts detection COCO format to panoptic COCO format. More
information about the formats can be found here:
http://cocodataset.org/#format-data.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import argparse
import numpy as np
import json
import time
import multiprocessing

# added these imports
from pycocotools import mask as COCOmask

import PIL.Image as Image

from panopticapi.utils import get_traceback, IdGenerator, save_json

try:
    # set up path for pycocotools
    # sys.path.append('./cocoapi-master/PythonAPI/')
    from pycocotools import mask as COCOmask
    from pycocotools.coco import COCO as COCO
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")

@get_traceback
def convert_detection_to_panoptic_coco_format_single_core(
    proc_id, coco_detection, img_ids, categories, segmentations_folder
):
    id_generator = IdGenerator(categories)

    annotations_panoptic = []
    for working_idx, img_id in enumerate(img_ids):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id,
                                                                 working_idx,
                                                                 len(img_ids)))
        img = coco_detection.loadImgs(int(img_id))[0]
        pan_format = np.zeros((img['height'], img['width'], 3))
        overlaps_map = np.zeros((img['height'], img['width']))

        anns_ids = coco_detection.getAnnIds(img_id)
        anns = coco_detection.loadAnns(anns_ids)

        panoptic_record = {}
        panoptic_record['image_id'] = img_id
        file_name = '{}.png'.format(img['file_name'].rsplit('.')[0])
        panoptic_record['file_name'] = file_name
        segments_info = []
        for ann in anns:
            if ann['category_id'] not in categories:
                raise Exception('Panoptic coco categories file does not contain \
                    category with id: {}'.format(ann['category_id'])
                )
            segment_id, color = id_generator.get_id_and_color(ann['category_id'])
            mask = coco_detection.annToMask(ann)
            overlaps_map += mask
            overlap_locations = (overlaps_map > 1)
            if np.sum(overlap_locations) != 0:
                overlaps_map -= mask
                mask[overlap_locations] = 0
                overlaps_map += mask

                # make mask with removed overlaps
                fortran_ground_truth_binary_mask = np.asfortranarray(mask)
                encoded_ground_truth = COCOmask.encode(fortran_ground_truth_binary_mask)
                ground_truth_area = COCOmask.area(encoded_ground_truth)
                ground_truth_bounding_box = COCOmask.toBbox(encoded_ground_truth)

                ann['bbox'] = ground_truth_bounding_box.tolist()
                ann['area'] = ground_truth_area.tolist()
            
            pan_format[mask == 1] = color
            ann.pop('segmentation')
            ann.pop('image_id')
            ann['id'] = segment_id
            segments_info.append(ann)

        if np.sum(overlaps_map > 1) != 0:
            raise Exception("Segments for image {} overlap each other.".format(img_id))

        # add nonarchaeo background semantic segmentation annotation
        ann['category_id'] = 2
        segment_id, color = id_generator.get_id_and_color(2)
        non_archaeo_mask = (1 - overlaps_map)
        fortran_ground_truth_binary_mask = np.asfortranarray(non_archaeo_mask, dtype=int)
        encoded_ground_truth = COCOmask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = COCOmask.area(encoded_ground_truth)
        ground_truth_bounding_box = COCOmask.toBbox(encoded_ground_truth)
        ann = segments_info[0].copy()
        ann['color'] = color
        ann['bbox'] = ground_truth_bounding_box.tolist()
        ann['area'] = ground_truth_area.tolist()
        pan_format[non_archaeo_mask == 1] = color
        ann['id'] = segment_id
        segments_info.append(ann)

        panoptic_record['segments_info'] = segments_info
        annotations_panoptic.append(panoptic_record)

        Image.fromarray(pan_format).save(os.path.join(segmentations_folder, file_name))

    print('Core: {}, all {} images processed'.format(proc_id, len(img_ids)))
    return annotations_panoptic


def convert_detection_to_panoptic_coco_format(input_json_file,
                                              segmentations_folder,
                                              output_json_file,
                                              categories_json_file):
    start_time = time.time()

    if segmentations_folder is None:
        segmentations_folder = output_json_file.rsplit('.', 1)[0]
    if not os.path.isdir(segmentations_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(segmentations_folder))
        os.mkdir(segmentations_folder)

    print("CONVERTING...")
    print("COCO detection format:")
    print("\tJSON file: {}".format(input_json_file))
    print("TO")
    print("COCO panoptic format")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(output_json_file))
    print('\n')

    coco_detection = COCO(input_json_file)
    img_ids = coco_detection.getImgIds()

    # removed dependence on categories file
    # archaeos and nonarchaeos
    categories_list = [{
            "id": 1,
            "name": "1.5",
            "supercategory": "",
            "isthing": "0",
            "color": [255,0,0],
            "metadata": {},
            "creator": "indannotate3",
            "keypoint_colors": []
        },
        {
            "id": 2,
            "name": "4.4",
            "supercategory": "",
            "isthing": "1",
            "color": [0,0,255],
            "metadata": {},
            "creator": "indannotate3",
            "keypoint_colors": []
        }]
    categories = {category['id']: category for category in categories_list}

    cpu_num = multiprocessing.cpu_count()
    img_ids_split = np.array_split(img_ids, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(img_ids_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, img_ids in enumerate(img_ids_split):
        p = workers.apply_async(convert_detection_to_panoptic_coco_format_single_core,
                                (proc_id, coco_detection, img_ids, categories, segmentations_folder))
        processes.append(p)
    annotations_coco_panoptic = []
    for p in processes:
        annotations_coco_panoptic.extend(p.get())

    with open(input_json_file, 'r') as f:
        d_coco = json.load(f)
    d_coco['annotations'] = annotations_coco_panoptic
    d_coco['categories'] = categories_list
    save_json(d_coco, output_json_file)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts detection COCO format to panoptic \
            COCO format. See this file's head for more information."
    )
    parser.add_argument('--input_json_file', type=str,
                        help="JSON file with detection COCO format")
    parser.add_argument('--output_json_file', type=str,
                        help="JSON file with panoptic COCO format")
    parser.add_argument(
        '--segmentations_folder', type=str, default=None, help="Folder with \
         panoptic COCO format segmentations. Default: X if output_json_file is \
         X.json"
    )
    parser.add_argument('--categories_json_file', type=str,
                        help="JSON file with Panoptic COCO categories information",
                        default='./panoptic_coco_categories.json')
    args = parser.parse_args()
    convert_detection_to_panoptic_coco_format(args.input_json_file,
                                              args.segmentations_folder,
                                              args.output_json_file,
                                              args.categories_json_file)
