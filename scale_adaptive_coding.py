import os
import json
import math
import numpy as np
import torch
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from generalized_RD_models import *
from bit_allocation import Bit_Allocation

from src.models.video_model import DMC
from src.models.image_model import IntraNoAR
from src.utils.common import str2bool, interpolate_log, create_folder, generate_log_json, dump_json
from src.utils.stream_helper import get_padding_size, get_state_dict
from src.utils.png_reader import PNGReader
from fe import FrameFusionNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
torch.set_num_threads(1)
np.random.seed(seed=0)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

up_tolerance = 0.2
# base_lambdas = [840, 380, 170, 85]
down_tolerance = 0.05

def rate_ratio_func(x, a, b):
    return a*(x**b)

def distortion_ratio_func(x, a, b):
    return a*(x**b)

def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image

def evaluate_one_video(gop, test_frame_nums, test_video_path, test_frame_height, test_frame_width, ratios, lambda_scales, mlmr_rate_popts, mlmr_distortion_popts, R_pic_avg_bpp, i_frame_net, video_net, base_lambda_idx, i_frame_weight, fe):       
    src_reader = PNGReader(test_video_path, test_frame_width, test_frame_height)
    
    sum_psnr=0.0
    sum_mse = 0.0
    sum_process_mse = 0.0
    bits = 0.0
    R_cur_gop_coded = 0.0
    # frame_nums = 0
    frame_pixel_num = test_frame_height * test_frame_width
    total_pixel_num = frame_pixel_num * test_frame_nums
    
    with torch.no_grad():
        for frame_idx in range(test_frame_nums):
            # up_tolerance = 0.5-0.12*(len(lambda_scales)-base_lambda_idx-1)+(0.5-0.12*(len(lambda_scales)-base_lambda_idx-1))*((frame_idx % gop) / gop)
            bit_allocation = Bit_Allocation(R_pic_avg_bpp, i_frame_weight)
            real_mse = 0.0
            real_bitrate = 0.0
            
            rgb = src_reader.read_one_frame(src_format="rgb")
            x = np_image_to_tensor(rgb)
            x = x.to(device)
            
            if frame_idx % gop == 0:
                # base_i_frame_scale = lambda_scales[base_lambda_idx][0]
                R_cur_gop_coded = 0.0
                R_gop = bit_allocation.gop_allocation(frame_idx, bits/frame_pixel_num, gop) # N_coded, R_coded, N_gop
                # R_pic = bit_allocation.pic_allocation(R_gop, R_cur_gop_coded, gop, frame_idx) #R_gop, R_cur_gop_coded, N_gop, frame_index

                # min_distortion = 10000.0
                # # min_rd_score = 10000.0
                # relative_ratio = 1.0
                # relative_rate_est = 10000.0
                # relative_lmbda_idx = 0

                # print("\n=======================frame idx "+str(frame_idx)+"========================")
                # for lmbda_idx in range(len(lambda_scales)):
                #     for ratio in ratios:
                #         rate_est = rate_ratio_func(ratio, mlmr_rate_popts[lmbda_idx][0], mlmr_rate_popts[lmbda_idx][1])
                #         print("rate_est: ", rate_est, "R_pic: ", R_pic)
                #         if 0 < rate_est <= (1 + up_tolerance) * R_pic:
                #             distortion = distortion_ratio_func(ratio, mlmr_distortion_popts[lmbda_idx][0], mlmr_distortion_popts[lmbda_idx][1])
                #             # rd_score = base_lambda*distortion - abs(rate_est-R_pic)
                #             if 0 < distortion < min_distortion:
                #             # if rd_score < min_rd_score:
                #                 min_distortion = distortion
                #                 # min_rd_score = rd_score
                #                 relative_lmbda_idx = lmbda_idx
                #                 relative_ratio = ratio
                #                 relative_rate_est = rate_est
                #                 # print("found point !")
                # print("=============================================================================")

                # downsampled_height = round(relative_ratio * test_frame_height)
                # downsampled_width = round(relative_ratio * test_frame_width)
                # padded_downsampled_width = downsampled_width if (downsampled_width % 64 == 0) else ((downsampled_width // 64 + 1) * 64)
                # padded_downsampled_height = downsampled_height if (downsampled_height % 64 == 0) else ((downsampled_height // 64 + 1) * 64)

                # x_down = F.interpolate(x, size=(downsampled_height, downsampled_width), mode='bicubic', align_corners=True).clamp(0.0, 1.0)

                padding_l, padding_r, padding_t, padding_b = get_padding_size(test_frame_height, test_frame_width)
                x_padded = torch.nn.functional.pad(
                    x,
                    (padding_l, padding_r, padding_t, padding_b),
                    mode="constant",
                    value=0,
                )
                
                # i_frame_scale, p_frame_y_q_scale, p_frame_mv_y_q_scale = lambda_scales[relative_lmbda_idx]

                result = i_frame_net.encode_decode(x_padded, 1.5409, None, pic_height=test_frame_height, pic_width=test_frame_width)
                dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                # recon_frame = result["x_hat"]
                bits += result["bit"]
                R_cur_gop_coded += (result["bit"] / frame_pixel_num)
                # frame_nums += (test_frame_height*test_frame_width)
                
            else:
                # R_gop = bit_allocation.gop_allocation(frame_idx, bits/frame_pixel_num, gop) # N_coded, R_coded, N_gop
                R_pic = bit_allocation.pic_allocation(R_gop, R_cur_gop_coded, gop, frame_idx) #R_gop, R_cur_gop_coded, N_gop, frame_index          
                
                min_distortion = 10000.0
                # min_rd_score = 10000.0
                relative_ratio = 0.5
                relative_rate_est = 10000.0
                relative_lmbda_idx = 3
                
                print("\n=======================frame idx "+str(frame_idx)+"========================")
                # Coding Parameter Determination
                for lmbda_idx in range(len(lambda_scales)):
                    for ratio in ratios:
                        rate_est = rate_ratio_func(ratio, mlmr_rate_popts[lmbda_idx][0], mlmr_rate_popts[lmbda_idx][1])
                        # print("rate_est: ", rate_est, "R_pic: ", R_pic)
                        if 0 < rate_est <= (1 + up_tolerance) * R_pic:
                            distortion = distortion_ratio_func(ratio, mlmr_distortion_popts[lmbda_idx][0], mlmr_distortion_popts[lmbda_idx][1])
                            print(mlmr_rate_popts[lmbda_idx], mlmr_distortion_popts[lmbda_idx], lmbda_idx, lambda_scales[lmbda_idx], ratio, "rate_est: ", rate_est, "R_pic: ", R_pic, "distortion: ", distortion, "min_distortion: ", min_distortion)
                            # rd_score = base_lambdas[len(base_lambdas)-base_lambda_idx-1]*distortion + rate_est
                            if 0 < distortion < min_distortion:
                            # if rd_score < min_rd_score:
                                min_distortion = distortion
                                # min_rd_score = rd_score
                                relative_lmbda_idx = lmbda_idx
                                relative_ratio = ratio
                                relative_rate_est = rate_est
                                # print("found point !")
                print("=============================================================================\n")
                
                _, p_frame_y_q_scale, p_frame_mv_y_q_scale = lambda_scales[relative_lmbda_idx]
                downsampled_height = round(relative_ratio * test_frame_height)
                downsampled_width = round(relative_ratio * test_frame_width)
                padded_downsampled_width = downsampled_width if (downsampled_width % 64 == 0) else ((downsampled_width // 64 + 1) * 64)
                padded_downsampled_height = downsampled_height if (downsampled_height % 64 == 0) else ((downsampled_height // 64 + 1) * 64)
                
                x_down = F.interpolate(x, size=(padded_downsampled_height, padded_downsampled_width), mode='bicubic', align_corners=True).clamp(0.0, 1.0)
                # pad if necessary
                # padding_l, padding_r, padding_t, padding_b = get_padding_size(downsampled_height, downsampled_width)
                # x_padded = torch.nn.functional.pad(
                #     x_down,
                #     (padding_l, padding_r, padding_t, padding_b),
                #     mode="constant",
                #     value=0,
                # )
                if dpb['ref_feature'] == None:
                    # print("before encoding: ", relative_ratio, frame_idx, x_down.shape, padded_downsampled_height, padded_downsampled_width)
                    dpb['ref_frame'] = F.interpolate(dpb['ref_frame'], size=(padded_downsampled_height, padded_downsampled_width), mode='bicubic', align_corners=True).clamp(0.0, 1.0)
                    # print("before encoding: ", frame_idx, x_down.shape, padded_downsampled_height, padded_downsampled_width, dpb['ref_frame'].shape)
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
                    # print("before encoding: ", relative_ratio, frame_idx, x_down.shape, padded_downsampled_height, padded_downsampled_width, dpb['ref_frame'].shape, dpb['ref_feature'].shape, dpb['ref_y'].shape)
                result = video_net.encode_decode(x_down, dpb, None,
                                                    pic_height=padded_downsampled_height, pic_width=padded_downsampled_width,
                                                    mv_y_q_scale=p_frame_mv_y_q_scale,
                                                    y_q_scale=p_frame_y_q_scale)
                dpb = result["dpb"]
                bits += result['bit']
                R_cur_gop_coded += (result["bit"] / frame_pixel_num)
            
            dpb["ref_frame"] = dpb["ref_frame"].clamp_(0, 1)
            # dpb["ref_frame"] = F.pad(dpb["ref_frame"], (-padding_l, -padding_r, -padding_t, -padding_b))
            if frame_idx % gop != 0:
                x_hat = F.interpolate(dpb["ref_frame"], size=(test_frame_height, test_frame_width), mode='bicubic', align_corners=True).clamp(0.0, 1.0)
            else:
                dpb["ref_frame"] = F.pad(dpb["ref_frame"], (-padding_l, -padding_r, -padding_t, -padding_b))
                x_hat = dpb["ref_frame"]

            processed_mse = (nn.MSELoss()(x, fe(x_hat))).item()
            
            real_mse = (nn.MSELoss()(x, x_hat)).item()
            real_bitrate = result['bit'] / frame_pixel_num
            
            # Model Updating
            if (frame_idx % gop != 0) and (relative_rate_est < 10000.0):
                print("\n", "frame_idx: ", frame_idx)
                print("lambda idx updated: ", relative_lmbda_idx, "relative ratio: ", relative_ratio)
                print("rate_popt: ", mlmr_rate_popts[relative_lmbda_idx], "distortion_popt: ", mlmr_distortion_popts[relative_lmbda_idx])
                print("real bitrate: ", real_bitrate, "real_mse: ", real_mse)
                print("relative_rate_est: ", relative_rate_est, "min_distortion: ", min_distortion)
                mlmr_rate_popts[relative_lmbda_idx][0] += 0.5 * (np.log(real_bitrate) - np.log(relative_rate_est)) * mlmr_rate_popts[relative_lmbda_idx][0]
                mlmr_rate_popts[relative_lmbda_idx][1] += 0.1 * (np.log(real_bitrate) - np.log(relative_rate_est)) * np.log(relative_ratio)
                mlmr_distortion_popts[relative_lmbda_idx][0] += 0.5 * (np.log(real_mse) - np.log(min_distortion)) * mlmr_distortion_popts[relative_lmbda_idx][0]
                mlmr_distortion_popts[relative_lmbda_idx][1] += 0.1 * (np.log(real_mse) - np.log(min_distortion)) * np.log(relative_ratio)
            
            psnr = -10 * np.log10(real_mse)
            sum_psnr += psnr
            sum_mse += real_mse
            sum_process_mse += processed_mse
        processed_final_mse = sum_process_mse / test_frame_nums
        final_mse = sum_mse / test_frame_nums
        final_psnr = sum_psnr / test_frame_nums
        final_bitrate = bits / total_pixel_num
        
        stats = {"test_sequence": test_video_path, "psnr": final_psnr, "mse": final_mse, "process_mse": processed_final_mse, "bitrate": final_bitrate, "est_curves": (mlmr_rate_popts, mlmr_distortion_popts)}

    return stats

def main():
    fe = FrameFusionNet(mid_channels=64, num_blocks=15, is_trainning=False)
    fe.cuda()
    # ratios = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    # init_ratios = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    ratios = [0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0]
    init_ratios = [0.5, 0.625, 0.75, 0.875, 1.0]
    scale_choices = [[0.5, 0.532, 0.919], [0.729, 0.713, 1.011], [1.083, 0.962, 1.104], [1.541, 1.238, 1.184]] # bpp逐渐减小
    # scale_choices = [[0.5, 0.532, 0.919], [0.626, 0.630, 0.967], [0.784, 0.746, 1.017], [0.982, 0.883, 1.070], [1.230, 1.046, 1.126], [1.541, 1.238, 1.184]]  # i_q_scale, p_y_q_scale, p_mv_y_q_scale
    # scale_choices = [[0.5, 0.5319, 0.9189], [0.729, 0.713, 1.011], [0.8469666666666666, 0.7673666666666666, 1.0074], [1.1939333333333333, 1.0028333333333332, 1.0958999999999999], [1.5409, 1.2383, 1.1844]]
    # scale_choices = [[0.5, 0.532, 0.919], [0.587, 0.600, 0.953], [0.690, 0.677, 0.988], [0.810, 0.764, 1.024], [0.951, 0.862, 1.062], [1.117, 0.973, 1.102], [1.312, 1.097, 1.142], [1.541, 1.238, 1.184]]

    # base_video_path = "/workspace/datasets/RL_video/RGB_1920x1080_64/"
    # seq_names = os.listdir(base_video_path)
    # for seq_name in seq_names:
    # video_path = os.path.join(base_video_path, seq_name)
    video_path = "/workspace/datasets/RL_video/BasketballDrive_1920x1080_50/"
    generalized_RD_Models = GeneralizedRateDistortionModels(init_ratios, scale_choices, video_path)
    mlmr_rate_popts, mlmr_distortion_popts = generalized_RD_Models.main_func()
    print("============Init Curve============>")
    print("mlmr_rate_popts:", mlmr_rate_popts)
    print("mlmr_distortion_popts:", mlmr_distortion_popts)
    print("<==================================")

    # 先拿eem不同downsample_ratio跑一个序列几个点，然后再拿这些点做限制码率，在附近搜索更好的点
    # test_bpps = [0.0176, 0.01791544117647059, 0.01599179602396514, 0.01480029511804722, 0.013285079656862745]
    # test_bpps = [0.014486249591059543, 0.007088699403638933, 0.004684018266989904, 0.003966209555373471]
    test_bpps = [0.032267, 0.045698, 0.070661, 0.114735]  # 0.000403  0.000338  0.000284  0.000249
    test_mses = [0.000403, 0.000338, 0.000284, 0.000249]
    # bit_allocation_method = 0
    # self.test_reference_points = {'point_0': [9.578001936461078e-05, 0.014486249591059543], 'point_1': [0.00010508783134355326, 0.007088699403638933], 'point_2': [0.00011670814842545951, 0.004684018266989904], 'point_3': [0.00012502443348694214, 0.003966209555373471]}

    for i in range(len(test_bpps)):
        if test_bpps[i] > 0.2:
            i_frame_weight = 5
        elif 0.1 < test_bpps[i] <= 0.2:
            i_frame_weight = 7
        else:
            i_frame_weight = 10
        results = evaluate_one_video(gop=32, test_frame_nums=96, test_video_path=video_path, test_frame_height=1080, test_frame_width=1920, ratios=ratios, lambda_scales=scale_choices, mlmr_rate_popts=mlmr_rate_popts, mlmr_distortion_popts=mlmr_distortion_popts, R_pic_avg_bpp=test_bpps[i], i_frame_net=generalized_RD_Models.i_frame_net, video_net=generalized_RD_Models.video_net, base_lambda_idx=i, i_frame_weight=i_frame_weight, fe=fe)
        print("===========final results===========>")
        print(results["test_sequence"], test_bpps[i], test_mses[i], results["psnr"], results["bitrate"],  results["mse"], results["process_mse"])
        print("est curves GR and GD: ", results["est_curves"])
        print("<===================================")

if __name__ == "__main__":
    main()