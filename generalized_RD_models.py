import os
import json
import math
import numpy as np
import torch
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F

from src.models.video_model import DMC
from src.models.image_model import IntraNoAR
from src.utils.common import str2bool, interpolate_log, create_folder, generate_log_json, dump_json
from src.utils.stream_helper import get_padding_size, get_state_dict
from src.utils.png_reader import PNGReader

from scipy import optimize as op

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
torch.set_num_threads(1)
np.random.seed(seed=0)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

class GeneralizedRateDistortionModels():
    def __init__(self, ratios, lambda_scales, seq_path, num_frames_initialize=4, gop=32, frame_width=1920, frame_height=1080):
        self.ratios = ratios
        self.lambda_scales = lambda_scales
        self.num_frames_initialize = num_frames_initialize
        self.gop = gop
        self.seq_path = seq_path
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.i_frame_model_path = "dummy_path_for_training/image_psnr.pth.tar"
        self.p_frame_model_path = "dummy_path_for_training/video_psnr.pth.tar"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        i_state_dict = get_state_dict(self.i_frame_model_path)
        self.i_frame_net = IntraNoAR()
        self.i_frame_net.load_state_dict(i_state_dict)
        self.i_frame_net = self.i_frame_net.to(self.device)
        self.i_frame_net.eval()
        
        p_state_dict = get_state_dict(self.p_frame_model_path)
        self.video_net = DMC()
        self.video_net.load_state_dict(p_state_dict)
        self.video_net = self.video_net.to(self.device)
        self.video_net.eval()

    def np_image_to_tensor(self, img):
        image = torch.from_numpy(img).type(torch.FloatTensor)
        image = image.unsqueeze(0)
        return image

    def PSNR(self, input1, input2):
        mse = torch.mean((input1 - input2) ** 2)
        psnr = 20 * torch.log10(1 / torch.sqrt(mse))
        return psnr.item()

    def enumerate_parameter_pairs(self):      
        multi_lmbda_multi_ratios_rate_points = []
        multi_lmbda_multi_ratios_distortion_points = []
        
        for scales in self.lambda_scales:
            i_frame_scale, p_frame_y_q_scale, p_frame_mv_y_q_scale = scales
            single_lmbda_multi_ratios_rate_points = []
            single_lmbda_multi_ratios_distortion_points = []

            for ratio in self.ratios:
                total_mse = 0.0
                bits = 0.0
                total_pixel_num = self.frame_height * self.frame_width * (self.num_frames_initialize - 1)
                src_reader = PNGReader(self.seq_path, self.frame_width, self.frame_height)
                
                with torch.no_grad():
                    # first_frame = (self.np_image_to_tensor(src_reader.read_one_frame(src_format="rgb"))).to(self.device)
                    # dpb = {
                    #     "ref_frame": first_frame,
                    #     "ref_feature": None,
                    #     "ref_y": None,
                    #     "ref_mv_y": None,
                    # }
                    # dpb["ref_frame"] = F.interpolate(dpb['ref_frame'], size=(round(0.9*self.frame_height), round(0.9*self.frame_width)), mode='bicubic', align_corners=True).clamp(0.0, 1.0)
                    # dpb["ref_frame"] = F.interpolate(dpb['ref_frame'], size=(self.frame_height, self.frame_width), mode='bicubic', align_corners=True).clamp(0.0, 1.0)  # pretend i frame encoded loss
                    for frame_idx in range(self.num_frames_initialize):
                        rgb = src_reader.read_one_frame(src_format="rgb")
                        x = self.np_image_to_tensor(rgb)
                        x = x.to(self.device)


                        if frame_idx % self.gop == 0:
                            # downsampled_height = round(ratio * self.frame_height)
                            # downsampled_width = round(ratio * self.frame_width)
                            # padded_downsampled_width = downsampled_width if (downsampled_width % 64 == 0) else ((downsampled_width // 64 + 1) * 64)
                            # padded_downsampled_height = downsampled_height if (downsampled_height % 64 == 0) else ((downsampled_height // 64 + 1) * 64)

                            # x_down = F.interpolate(x, size=(downsampled_height, downsampled_width), mode='bicubic', align_corners=True).clamp(0.0, 1.0)

                            # pad if necessary
                            padding_l, padding_r, padding_t, padding_b = get_padding_size(self.frame_height, self.frame_width)
                            x_padded = torch.nn.functional.pad(
                                x,
                                (padding_l, padding_r, padding_t, padding_b),
                                mode="constant",
                                value=0,
                            )

                            result = self.i_frame_net.encode_decode(x_padded, 1.5409, None,
                                                    pic_height=self.frame_height, pic_width=self.frame_width)
                            dpb = {
                                "ref_frame": result["x_hat"],
                                "ref_feature": None,
                                "ref_y": None,
                                "ref_mv_y": None,
                            }
                            # bits += result["bit"]
                            # print(scales, ratio, "i frame costed", result["bit"] / (self.frame_height * self.frame_width))                           
                        else:
                            downsampled_height = round(ratio * self.frame_height)
                            downsampled_width = round(ratio * self.frame_width)
                            padded_downsampled_width = downsampled_width if (downsampled_width % 64 == 0) else ((downsampled_width // 64 + 1) * 64)
                            padded_downsampled_height = downsampled_height if (downsampled_height % 64 == 0) else ((downsampled_height // 64 + 1) * 64)
                    
                            x_down = F.interpolate(x, size=(padded_downsampled_height, padded_downsampled_width), mode='bicubic', align_corners=True).clamp(0.0, 1.0)
                            # padding_l, padding_r, padding_t, padding_b = get_padding_size(downsampled_height, downsampled_width)
                            # x_padded = torch.nn.functional.pad(
                            #     x_down,
                            #     (padding_l, padding_r, padding_t, padding_b),
                            #     mode="constant",
                            #     value=0,
                            # )
                            if dpb['ref_feature'] == None:
                                dpb['ref_frame'] = F.interpolate(dpb['ref_frame'], size=(padded_downsampled_height, padded_downsampled_width), mode='bicubic', align_corners=True).clamp(0.0, 1.0)
                                # dpb['ref_frame'] = torch.nn.functional.pad(
                                #     dpb['ref_frame'],
                                #     (padding_l, padding_r, padding_t, padding_b),
                                #     mode="constant",
                                #     value=0,
                                # )
                            else:
                                dpb['ref_frame'] = F.interpolate(dpb['ref_frame'], size=(padded_downsampled_height, padded_downsampled_width), mode='bicubic', align_corners=True).clamp(0.0, 1.0)
                                # dpb['ref_frame'] = torch.nn.functional.pad(
                                #     dpb['ref_frame'],
                                #     (padding_l, padding_r, padding_t, padding_b),
                                #     mode="constant",
                                #     value=0,
                                # )
                                dpb['ref_feature'] = F.interpolate(dpb['ref_feature'], size=(padded_downsampled_height, padded_downsampled_width), mode='bicubic', align_corners=True)  # 应该是不用clamp
                                dpb['ref_y'] = F.interpolate(dpb['ref_y'], size=(padded_downsampled_height//16, padded_downsampled_width//16), mode='bicubic', align_corners=True)
                                dpb['ref_mv_y'] = F.interpolate(dpb['ref_mv_y'], size=(padded_downsampled_height//16, padded_downsampled_width//16), mode='bicubic', align_corners=True)
                                
                            result = self.video_net.encode_decode(x_down, dpb, None,
                                                    pic_height=padded_downsampled_height, pic_width=padded_downsampled_width,
                                                    mv_y_q_scale=p_frame_mv_y_q_scale,
                                                    y_q_scale=p_frame_y_q_scale)
                            dpb = result["dpb"]
                            bits += result['bit']

                        dpb["ref_frame"] = dpb["ref_frame"].clamp_(0, 1)
                        # dpb["ref_frame"] = F.pad(dpb["ref_frame"], (-padding_l, -padding_r, -padding_t, -padding_b))
                        # if frame_idx % self.gop != 0:
                        
                        if frame_idx % self.gop != 0:
                            x_hat = F.interpolate(dpb["ref_frame"], size=(self.frame_height, self.frame_width), mode='bicubic', align_corners=True).clamp(0.0, 1.0)
                            total_mse += (nn.MSELoss()(x, x_hat)).item()
                        else:
                            dpb["ref_frame"] = F.pad(dpb["ref_frame"], (-padding_l, -padding_r, -padding_t, -padding_b))
                            x_hat = dpb["ref_frame"]
                            # total_mse += (nn.MSELoss()(x, x_hat)).item()
                            # total_mse += (nn.MSELoss()(x, x_hat)).item()
                            # else:
                            #     x_hat = dpb["ref_frame"]
                                # total_mse += (nn.MSELoss()(x, x_hat)).item()

                            
                single_lmbda_multi_ratios_rate_points.append(bits / total_pixel_num)
                single_lmbda_multi_ratios_distortion_points.append((total_mse / (self.num_frames_initialize - 1)))
            multi_lmbda_multi_ratios_rate_points.append(single_lmbda_multi_ratios_rate_points)
            multi_lmbda_multi_ratios_distortion_points.append(single_lmbda_multi_ratios_distortion_points)
            # torch.cuda.empty_cache()

        return multi_lmbda_multi_ratios_rate_points, multi_lmbda_multi_ratios_distortion_points
    
    def target_func(self, x, a, b):
        return a*(x**b)
    
    def curve_output(self, rate_points, distortion_points): #fitting rate_ratio curves / distortion_ratio curves
        rate_values = rate_points
        distortion_values = distortion_points
        single_lmbda_multi_ratio_popt_rate, _ = op.curve_fit(self.target_func, self.ratios, rate_values)
        single_lmbda_multi_ratio_popt_distortion, _ = op.curve_fit(self.target_func, self.ratios, distortion_values)
        return single_lmbda_multi_ratio_popt_rate, single_lmbda_multi_ratio_popt_distortion
    
    def main_func(self): #main function
        mlmr_rate_points, mlmr_distortion_points = self.enumerate_parameter_pairs()
        print("mlmr rate points: ", mlmr_rate_points)
        print("mlmr distortion points: ", mlmr_distortion_points)
        mlmr_rate_popts = []
        mlmr_distortion_popts = []
        for idx in range(len(mlmr_rate_points)):
            rate_popt_idx, distortion_popt_idx = self.curve_output(mlmr_rate_points[idx], mlmr_distortion_points[idx])
            mlmr_rate_popts.append(rate_popt_idx)
            mlmr_distortion_popts.append(distortion_popt_idx)
        return mlmr_rate_popts, mlmr_distortion_popts  #a1,b1  a2,b2     
        
