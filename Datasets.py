from mrcnn import utils
import numpy as np
from xml.dom import minidom
import os
import skimage
import json


class ParkingLotDataset(utils.Dataset):

    def load_parkinglot(self, file_list):
        self.add_class("empty", 1, "empty")
        self.add_class("occupied", 2, "occupied")

        for file in file_list:
            image_name = os.path.splitext(file)[0]+'.jpg'
            doc = minidom.parse(file)
            space_list = doc.getElementsByTagName('space')
            image = skimage.io.imread(image_name)
            height, width = image.shape[:2]
            self.add_image(
                "empty",
                image_id=file,
                path=image_name,
                width=width, height=height,
                polygons=space_list)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "empty":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        classes = []
        for i, p in enumerate(info["polygons"]):
            points = p.getElementsByTagName("point")
            if len(points) == 0:
                points = p.getElementsByTagName("Point")
            x, y = [], []
            for point in points:
                y.append(point.attributes['y'].value)
                x.append(point.attributes['x'].value)
            y = np.array(y).astype(np.int32)
            x = np.array(x).astype(np.int32)
            rr, cc = skimage.draw.polygon(y, x)
            if p.hasAttribute('occupied'):
                classes.append(p.attributes['occupied'].value)
            else:
                classes.append("0")
            mask[rr, cc, i] = 1

        return mask.astype(np.bool), (np.array(classes, dtype=np.int32) + 1)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "empty":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class ParkingLotManualDataset(utils.Dataset):

    def load_parkinglot(self, dataset_dir):
        self.add_class("lot", 1, "lot")
        self.add_class("cars", 2, "cars")

        annotations = json.load(open(os.path.join(dataset_dir, "via_parkinglot.json")))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            annotations = [r['region_attributes'] for r in a['regions']]
            self.add_image(
                "lot",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                annotations=annotations)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "lot":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        classes = []
        for i, p in enumerate(info["annotations"]):
            classes.append(p['class'])
        return mask.astype(np.bool), (np.array(classes, dtype=np.int32) + 1)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "lot":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
