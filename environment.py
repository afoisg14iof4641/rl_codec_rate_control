import os
import json
import math
import numpy as np
import torch
from PIL import Image
import cv2
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader

from codec_model import Codec_Model

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
random.seed(42)
torch.manual_seed(42)
np.random.seed(seed=0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_si(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    si = np.std(grad_magnitude)
    return si


def calculate_ti(frame1, frame2):
    assert frame1.shape == frame2.shape
    diff_squared = (frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2
    ti = np.sqrt(np.mean(diff_squared))
    return ti


class Video_Dataset(data.Dataset):
    def __init__(self, frame_num=32, random_choice=False, random_rot=False, max_random_frame_range=2):
        self.images_list = []
        self.num = 0
        self.frame_num = frame_num
        self.random_choice = random_choice
        self.max_random_frame_range = max_random_frame_range

        trainset_path = "/workspace/datasets/RL_video/RGB/"
        all_seqs = os.listdir(trainset_path)
        seqs = [seq for seq in all_seqs if seq.startswith("B") == True or seq.startswith("C") == True or seq.startswith("D") == True]
        for seq in seqs:
            for poc in range(1, 65-max_random_frame_range*frame_num, frame_num):
                frame_list = []
                if not self.random_choice:
                    for poc_idx in range(self.frame_num):
                        cur_pic = trainset_path + seq + "/" + str(poc_idx+poc).rjust(8, "0") + ".png"
                        frame_list.append(cur_pic)
                else:
                    poc_idx = 0
                    while poc_idx < frame_num:
                        step = random.choice(list(range(0, max_random_frame_range)))
                        cur_pic = trainset_path + seq + "/" + str(poc_idx+poc+step).rjust(8, "0") + ".png"
                        frame_list.append(cur_pic)
                        poc_idx += 1
                    assert len(frame_list) == frame_num
                self.images_list += [frame_list]
                self.num = self.num + 1
        
        print('Prepared training data nums', self.num)

    def get_images_list(self, shuffle=True):
        random.shuffle(self.images_list)
        return self.images_list
    
    def __len__(self):
        return self.num

class Encoder_and_Decoder(object):
    def __init__(self, lmb_range=(256.0, 2048.0), gop=32, from_json=False):
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        self.lmb_range = lmb_range

        self.occupied_R = 0.0
        self.occupied_D = 0.0
        self.remaining_D = 0.0
        self.remaining_R = 1.0

        self.D_decay = 1.0
        self.R_decay = 1.0
        
        # self.reward_list = [0, 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22, 25]
        # self.punish_list = [-22, -20, -18, -16, -14, -12, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]
        
        self.target_R = 0.0
        self.target_D = 0.0
        self.base_lmb = 1.0
        if from_json:
            with open("dummy_path_for_paper/target.json", "r") as ftar:
                self.target_dict = json.load(ftar)
        else:
            self.target_dict = None
        
        train_dataset = Video_Dataset(frame_num=32)
        self.train_sequences = train_dataset.get_images_list()
        self.current_video_name = None
        self.test_sequences_path = "/workspace/datasets/USTC-TD_video_png/"
        self.test_sequences = os.listdir(self.test_sequences_path)
        self.test_action = {}
        for seq in self.test_sequences:
            self.test_action[seq] = {}
            for fi in range(gop):
                self.test_action[seq]["frame_"+str(fi)] = {}
                self.test_action[seq]["frame_"+str(fi)]["cur_action"] = 0.0
        
        self.gop = gop
        # self.framenums = 64
        self.current_frame = 0
        self.current_video = 0
        
        # self.width = width
        # self.height = height
        
        self.net = Codec_Model()
        snapshot = torch.load('dummy_path_for_paper/refine_psnr.pth.tar')
        self.net.load_state_dict(snapshot['state_dict'])
        self.net.to(device).eval()
        
        self.temp_lists = []
        self.ref = None
        self.cur = None

    def reset(self):
        state = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        over = 0
        self.occupied_D = 0.0
        self.occupied_R = 0.0
        self.remaining_D = 1.0
        self.remaining_R = 1.0
        self.ref = None
        self.cur = None
        self.temp_lists = []
        self.target_R = 0.0
        self.target_D = 0.0
        self.base_lmb = 1.0
        return state, over
    
    def pad(self, x, p=2 ** 6):
        h, w = x.size(2), x.size(3)
        H = (h + p - 1) // p * p
        W = (w + p - 1) // p * p
        padding_left = (W - w) // 2
        padding_right = W - w - padding_left
        padding_top = (H - h) // 2
        padding_bottom = H - h - padding_top
        return F.pad(
            x,
            (padding_left, padding_right, padding_top, padding_bottom),
            mode="constant",
            value=0,
        )

    def crop(self, x, size):
        H, W = x.size(2), x.size(3)
        h, w = size
        padding_left = (W - w) // 2
        padding_right = W - w - padding_left
        padding_top = (H - h) // 2
        padding_bottom = H - h - padding_top
        return F.pad(
            x,
            (-padding_left, -padding_right, -padding_top, -padding_bottom),
            mode="constant",
            value=0,
        )
    
    def get_base_point(self, mode="train"):
        if self.target_dict != None:
            self.target_R = self.target_dict[self.current_video]["target_bitrate"]
            self.target_D = self.target_dict[self.current_video]["target_distortion"]
            self.base_lmb = self.target_dict[self.current_video]["base_lambda"]
        else:
            self.base_lmb = 256.0 + random.random() * 1792.0
            sum_mse = sum_bpp = 0.0
            for frame_idx in range(self.gop):
                if mode == 'train':
                    imgs_path = self.train_sequences[self.current_video][self.current_frame]
                else:
                    imgs_path = os.path.join(self.test_sequences_path, self.test_sequences[self.current_video], "im"+str(frame_idx+1).rjust(5, "0")+".png")

                with torch.no_grad():
                    cur = self.read_image(imgs_path)
                    x_pad = self.pad(cur, 64)
                    lmb = torch.full((1, ), self.base_lmb).to(device)
                    
                    if frame_idx % self.gop == 0:
                        temp_list = self.net.get_temp_bias(x_pad)
                        temp_lists = [temp_list, temp_list]
                        ref = None
                    
                    frame_metrics, fdict = self.net.forward_frame(x_pad, ref, temp_lists, lmb=lmb, return_fdict=True)

                    ref = fdict["x_hat"].clamp_(0, 1)
                    temp_lists = [temp_lists[-1], fdict["temp_new_list"]]

                    mse_loss = torch.mean((cur-self.crop(ref, (cur.size(2), cur.size(3))))**2).item()
                    bpp_loss = frame_metrics["bpp"]
                    sum_mse += mse_loss
                    sum_bpp += bpp_loss
            
            self.target_R = sum_bpp / self.gop
            self.target_D = sum_mse / self.gop

            
    def record_actions(self, action):
        self.current_video_name = self.train_sequences[self.current_video][0].split("/")[-2] + "_" + self.train_sequences[self.current_video][0].split("/")[-1].split(".")[0].lstrip("0")
        if self.current_video == 0 and self.current_frame == 0:
            actions_dict = {}
            current_action_dict = {}
            current_action_dict["cur_action"] = action
            # current_action_dict["current_action_choice"] = action
            actions_dict[self.current_video_name] = {}
            actions_dict[self.current_video_name]["frame_"+str(self.current_frame)] = current_action_dict
            actions_dict[self.current_video_name]["base_lmb"] = self.base_lmb
            
            with open("dummy_path_for_paper/actions.json", "w") as fact:
                fact.write(json.dumps(actions_dict, indent=2))
        else:
            with open("dummy_path_for_paper/actions.json", "r") as fread:
                actions_dict = json.load(fread)
            current_action_dict = {}
            current_action_dict["cur_action"] = action
            if self.current_frame == 0:
                video_actions_dict = {}
                video_actions_dict["frame_"+str(self.current_frame)] = current_action_dict
                video_actions_dict["base_lmb"] = self.base_lmb
                actions_dict[self.current_video_name] = video_actions_dict
            else:
                actions_dict[self.current_video_name]["frame_"+str(self.current_frame)] = current_action_dict
                actions_dict[self.current_video_name]["base_lmb"] = self.base_lmb
            
            with open("dummy_path_for_paper/actions.json", "w") as fact:
                fact.write(json.dumps(actions_dict, indent=2))
            
    def get_observation(self, mode='train'):
        if mode == 'train' or mode == "init":
            # mean std msad
            if self.ref == None or self.temp_lists == [] or self.cur == None:
                ref_mean = ref_std = cur_mean = cur_std = temp_list_idx0_mean = temp_list_idx0_std = temp_list_idx1_mean = temp_list_idx1_std = 0.0
            else:
                ref_mean = torch.mean(self.ref).item()
                ref_std = torch.std(self.ref).item()
                cur_mean = torch.mean(self.cur).item()
                cur_std = torch.std(self.cur).item()
                temp_list_idx0_mean = np.mean([torch.mean(temp_list).item() for temp_list in self.temp_lists[0]])
                temp_list_idx0_std = np.mean([torch.std(temp_list).item() for temp_list in self.temp_lists[0]])
                temp_list_idx1_mean = np.mean([torch.mean(temp_list).item() for temp_list in self.temp_lists[1]])
                temp_list_idx1_std = np.mean([torch.std(temp_list).item() for temp_list in self.temp_lists[1]])
            with open("dummy_path_for_paper/actions.json", "r") as fread_act:
                actions = json.load(fread_act)
            last_action = actions[self.current_video_name]["frame_"+str(self.current_frame)]
        else:
            # mean std msad
            if self.ref == None or self.temp_lists == [] or self.cur == None:
                ref_mean = ref_std = cur_mean = cur_std = temp_list_idx0_mean = temp_list_idx0_std = temp_list_idx1_mean = temp_list_idx1_std = 0.0
            else:
                ref_mean = torch.mean(self.ref).item()
                ref_std = torch.std(self.ref).item()
                cur_mean = torch.mean(self.cur).item()
                cur_std = torch.std(self.cur).item()
                temp_list_idx0_mean = np.mean([torch.mean(temp_list).item() for temp_list in self.temp_lists[0]])
                temp_list_idx0_std = np.mean([torch.std(temp_list).item() for temp_list in self.temp_lists[0]])
                temp_list_idx1_mean = np.mean([torch.mean(temp_list).item() for temp_list in self.temp_lists[1]])
                temp_list_idx1_std = np.mean([torch.std(temp_list).item() for temp_list in self.temp_lists[1]])
            last_action = self.test_action[self.current_video_name]["frame_"+str(self.current_frame)]

        self.remaining_R = 1.0 - self.occupied_R / (self.target_R * self.R_decay * 32.0)
        self.remaining_D = 1.0 - self.occupied_D / (self.target_D * self.D_decay * 32.0)
        
        poc = 1.0-((self.current_frame+1)%32)/31.0

        # without remaining D for bit allocation
        return [last_action["cur_action"], poc, self.remaining_R, cur_mean, cur_std, ref_mean, ref_std, temp_list_idx0_mean, temp_list_idx0_std, temp_list_idx1_mean, temp_list_idx1_std]
    
    def select_action(self, action):
        return action
    
    def read_image(self, ori_fp):
        x = cv2.cvtColor(cv2.imread(str(ori_fp)), cv2.COLOR_BGR2RGB) / 255.
        x = torch.FloatTensor(x).permute(2, 0, 1).unsqueeze(0).to(device)
        
        return x
        
    def step(self, action, mode='train'):
        # print(len(self.train_sequences))
        if mode == 'train' or mode == "init":
            imgs_path = self.train_sequences[self.current_video][self.current_frame]
            action = self.select_action(action)
            self.record_actions(action)
        else:
            imgs_path = os.path.join(self.test_sequences_path, self.test_sequences[self.current_video], "im"+str(self.current_frame+1).rjust(5, "0")+".png")
            action = self.select_action(action)
            self.current_video_name = self.test_sequences[self.current_video]
            self.test_action[self.current_video_name]["frame_"+str(self.current_frame)]["cur_action"] = action

        with torch.no_grad():
            self.cur = self.read_image(imgs_path)
            x_pad = self.pad(self.cur, 64)
            lmb = torch.full((1, ), 256.0+action*1792.0).to(device)

            if self.current_frame % self.gop == 0:
                temp_list = self.net.get_temp_bias(x_pad)
                self.temp_lists = [temp_list, temp_list]
                self.ref = None
            
            frame_metrics, fdict = self.net.forward_frame(x_pad, self.ref, self.temp_lists, lmb=lmb, return_fdict=True)

            self.ref = fdict["x_hat"].clamp_(0, 1)
            self.temp_lists = [self.temp_lists[-1], fdict["temp_new_list"]]

            mse_loss = torch.mean((self.cur-self.crop(self.ref, (self.cur.size(2), self.cur.size(3))))**2).item()
            bpp_loss = frame_metrics["bpp"]

            self.occupied_D += mse_loss
            self.occupied_R += bpp_loss

        if mode == 'train' or mode == "init":
            if self.current_frame % self.gop == 0:
                self.get_base_point(mode='train')
            state = self.get_observation(mode='train')
            self.current_frame += 1                           
            if self.current_frame == self.gop:
                self.current_video += 1
                self.current_frame = 0
                over = 1
            else:
                over = 0
            if self.current_video >= len(self.train_sequences):
                self.current_video = 0

            reward = -1 * (self.base_lmb * mse_loss + bpp_loss)
            if self.current_frame == 0:
                # if self.remaining_D >= 0:
                #     reward += (100 * self.remaining_D)
                if self.remaining_R != 0:
                    reward -= (1000 * abs(self.remaining_R))

            # without target D and remaining D
            if self.current_video % 80 == 0:
                print(f"Train Video {self.current_video_name}, Frame {self.current_frame}")
                print("<= preset =>", self.target_R, self.base_lmb)
                print("<= codec =>", mse_loss, bpp_loss)
                print("<= action =>", action, self.remaining_R, reward)
                print("===")
            
            return state, reward, over
        else:
            if self.current_frame % self.gop == 0:
                self.get_base_point(mode='test')
            state = self.get_observation(mode='test')
            self.current_frame += 1                           
            if self.current_frame == self.gop:
                self.current_video += 1
                self.current_frame = 0
                over = 1
            else:
                over = 0
                
            if self.current_video >= len(self.test_sequences):
                self.current_video = 0
                
            reward = -1 * (self.base_lmb * mse_loss + bpp_loss)
            if self.current_frame == 0:
                # if self.remaining_D >= 0:
                #     reward += (100 * self.remaining_D)
                if self.remaining_R != 0:
                    reward -= (1000 * abs(self.remaining_R))

            # without target D and remaining D
            print(f"Val Video {self.current_video_name}, Frame {self.current_frame}")
            print("<= preset =>", self.target_R, self.base_lmb)
            print("<= codec =>", mse_loss, bpp_loss)
            print("<= action =>", action, self.remaining_R, reward)
            print("===")

            return state, reward, over, mse_loss, bpp_loss
