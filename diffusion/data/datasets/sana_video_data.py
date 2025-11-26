import getpass
import json
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from termcolor import colored
from torch.utils.data import Dataset
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
from .read_video import read_video
from diffusion.data.builder import DATASETS, get_data_path
from diffusion.data.wids import ShardListDataset, ShardListDatasetMulti, lru_json_load
from diffusion.utils.logger import get_root_logger
from .utils import read_file, get_transforms_image, get_transforms_video, temporal_random_crop
from torchvision.io import write_video


@DATASETS.register_module()
class VideoTextDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path=None,
        num_frames=16,
        frame_interval=1,
        #image_size=(256, 256),
        image_height=256,
        image_width=256,
        transform_name="center",
        caption_proportion=None,
        external_caption_suffixes=None,
        external_clipscore_suffixes=None,
        clip_thr=0.0,
        clip_thr_temperature=1.0,
        **kwargs,
    ):
        self.data_path = data_path
        self.data = read_file(data_path[0])
        
        # self.data = data_csv(data_path)
        self.get_text = "text" in self.data.columns
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = (image_height,image_width)
        self.transforms = {
            "image": get_transforms_image(transform_name, self.image_size),
            "video": get_transforms_video(transform_name, self.image_size),
        }
        self.ori_imgs_nums = len(self)

    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["video_path"]
        flow_score = sample["flow"]
        file_type = self.get_type(path)
        if file_type == "video":
            # loading
            vframes, vinfo = read_video(path, backend="av")
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            # video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)
            video = vframes
            # transform
            
            #print(video.shape)
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
            #print(video.shape)

        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = self.transforms["image"]
            image = transform(image)

            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        ret = {"video": video, "fps": video_fps, "height": self.image_size, "width": self.image_size, "num_frames": self.num_frames, "ar": 1, "flow_score":flow_score, "path": path}
        if self.get_text:
            ret["text"] = sample["text"]
        return ret

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)