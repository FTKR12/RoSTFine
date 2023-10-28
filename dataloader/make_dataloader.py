import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import math

from PIL import Image
import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import logging

log = logging.getLogger(__name__)

def LoadData(args):

    with open(F"{args.data_dir}/{args.grade_path}", mode="rb") as f:
        grade_dict = pickle.load(f)    
    with open(F"{args.data_dir}/{args.video_path}", mode="rb") as f:
        video_dict = pickle.load(f)
    with open(F"{args.data_dir}/{args.traj_path}", mode="rb") as f:
        traj_dict = pickle.load(f)
    
    sperm_ids = np.array(list(grade_dict.keys()))

    log.info(f'[DATA] (grade) {len(grade_dict)}    (video) {len(video_dict)}    (traj) {len(traj_dict)}')
    return sperm_ids, grade_dict, video_dict, traj_dict

class PackPathway(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.slowfast_alpha = 4
        
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1]//self.slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

#データセット作成
class SpermDataset(Dataset):
    # structure of SpermDataset
    def __init__(self, videos, trajs, grades, IDs, num_frame, video_size=150, pathway=False):
        self.videos = videos
        self.grades = grades
        self.trajs = trajs
        self.IDs = IDs
        self.num_frame = num_frame
        self.video_size = video_size
        self.pathway = pathway

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, i):
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        if self.pathway != 'slowfast':
            transform = Compose(
                [
                    UniformTemporalSubsample(self.num_frame)
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=self.video_size
                    )
                ]
            )
        
        else:
            transform = Compose(
                [
                    UniformTemporalSubsample(self.num_frame),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=self.video_size
                    ),
                    PackPathway()
                ]
            ) 
            
        video = torch.from_numpy(self.videos[i].astype(np.float32)).permute(3,0,1,2)
        video = transform(video)

        grade = torch.Tensor(self.grades[i])
        traj = torch.Tensor(np.array(self.trajs[i]))
        ID = str(self.IDs[i])
        return video, traj, grade, ID

def make_kfold_dataloader(k, sperm_ids, video_dict, traj_dict, grade_dict, num_frame=8, video_size=224, batch_size=32, pathway=False):
    kf = KFold(n_splits=k, shuffle=True, random_state=123)

    kfold_id_train = []          
    kfold_id_test = []            
    kfold_train_dataloader_learn = []  
    kfold_test_dataloader = []        

    for index_train, index_test in kf.split(sperm_ids):
        train_id = sperm_ids[index_train]
        test_id = sperm_ids[index_test]

        train_video = np.array([video_dict[ID] for ID in train_id])
        train_traj = np.array([traj_dict[ID] for ID in train_id])

        train_grade = np.array([grade_dict[ID] for ID in train_id])

        test_video = np.array([video_dict[ID] for ID in test_id])
        test_traj = np.array([traj_dict[ID] for ID in test_id])
     
        test_grade = np.array([grade_dict[ID] for ID in test_id])

        train_dataset = SpermDataset(videos=train_video, trajs=train_traj, grades=train_grade, IDs=train_id, num_frame=num_frame, video_size=video_size, pathway=pathway)
        test_dataset = SpermDataset(videos=test_video, trajs=test_traj, grades=test_grade, IDs=test_id, num_frame=num_frame, video_size=video_size, pathway=pathway)

        train_dataloader_learn = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        kfold_train_dataloader_learn.append(train_dataloader_learn)
        kfold_test_dataloader.append(test_dataloader)

    return kfold_train_dataloader_learn, kfold_test_dataloader