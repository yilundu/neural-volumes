# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import imageio

from PIL import Image
import os.path as osp

import torch.utils.data
import json
import os

def load_krt(path):
    """Load KRT file containing intrinsic and extrinsic parameters."""
    meta = json.load(open(osp.join(path, 'transforms_train.json'), "r"))
    frames = meta['frames']
    timesteps = {}

    for frame in frames:
        timestep = frame['timestep']
        filenames = timesteps.get(timestep, [])
        filenames.append(frame["file_path"])
        timesteps[timestep] = filenames

    for k, v in timesteps.items():
        v.sort()

    camera_angle_x = meta['camera_angle_x']

    return frames, camera_angle_x, timesteps

class Dataset(torch.utils.data.Dataset):
    def __init__(self, full=False):
        krtpath = "experiments/pouring/pouring_dataset_large/"
        krt, camera_angle_x, timesteps = load_krt(krtpath)
        self.timesteps = timesteps
        self.meta = krt
        self.full = full
        self.camera_angle = camera_angle_x
        self.basedir = krtpath
        self.subsamplesize = 128
        self.cameras = []
        self.imagemean = 100
        self.imagestd = 25
        self.keyfilter = []
        self.fixedcameras = [1] * 3
        self.transf = np.eye(3) * 0.4

    def get_allcameras(self):
        return []

    def get_krt(self):
        return {k: {
                "pos": self.campos[k],
                "rot": self.camrot[k],
                "focal": self.focal[k],
                "princpt": self.princpt[k],
                "size": np.array([667, 1024])}
                for k in self.cameras}


    def known_background(self):
        return "bg" in self.keyfilter

    def get_background(self, bg):
        if "bg" in self.keyfilter:
            for i, cam in enumerate(self.cameras):
                if cam in self.bg:
                    bg[cam].data[:] = torch.from_numpy(self.bg[cam]).to("cuda")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        frame = self.meta[idx]
        fname = os.path.join(self.basedir, frame['file_path'])
        pose = np.array(frame['transform_matrix'])
        timestep = frame['timestep']

        result = {}

        ninput = len(self.fixedcameras)

        fixedcamimage = np.zeros((3 * ninput, 400, 400), dtype=np.float32)
        for i in range(ninput):
            imagepath = self.timesteps[timestep][i]
            imagepath = osp.join(self.basedir, imagepath)
            image = np.asarray(Image.open(imagepath), dtype=np.uint8).transpose((2, 0, 1)).astype(np.float32) / 255.
            if np.sum(image) == 0:
                validinput = False
            fixedcamimage[i*3:(i+1)*3, :, :] = image[:3]

        image = np.array(imageio.imread(fname)[:, :, :3]).transpose((2, 0, 1))
        C, H, W = image.shape
        self.focal = .5 * W / np.tan(.5 * self.camera_angle)
        fixedcamimage[:] -= self.imagemean
        fixedcamimage[:] /= self.imagestd
        result["fixedcamimage"] = fixedcamimage

        result['camrot'] = np.dot(self.transf, pose[:3, :3].astype(np.float32)).T
        result['campos'] = np.dot(self.transf, pose[:3, 3].astype(np.float32))
        result["image"] = image.astype(np.float32)

        result["focal"] = np.array([self.focal]).astype(np.float32)
        result["princpt"] = np.array((H//2, H//2)).astype(np.float32)
        result["camindex"] = 0
        validinput = True
        result["validinput"] = np.float32(1.0 if validinput else 0.0)
        # camera data


        if self.full:
            px, py = np.meshgrid(np.arange(W).astype(np.float32), np.arange(H).astype(np.float32))
        else:
            px = np.random.randint(0, W, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
            py = np.random.randint(0, H, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)

        result["pixelcoords"] = np.stack((px, py), axis=-1).astype(np.float32)

        return result
