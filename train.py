import os
import torch
import random
import numpy as np
import argparse
import torch.nn.functional as F

from environment import Encoder_and_Decoder
from rl_network import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.distributions import Normal

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
random.seed(42)
torch.manual_seed(42)
np.random.seed(seed=42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad_(value)

def soft_update(_from, _to):
    for _from, _to in zip(_from.parameters(), _to.parameters()):
        value = _to.data * 0.995 + _from.data * 0.005
        _to.data.copy_(value)
        
def encode_one_video(model_action, env, mode='train', norm_mean=[], norm_std=[]):
    data = []
    reward_sum = 0.0
    video_mse = 0.0
    video_bitrate = 0.0

    last_state, over = env.reset()

    for i in range(env.gop):
        mu_list, sigma_list = model_action(torch.tensor(last_state, dtype=torch.float32, device=device).reshape(-1, state_dim))
        action= []
        for action_dim in range(len(mu_list)):
            action.append(1 / (1 + np.exp(-1*random.normalvariate(mu=mu_list[action_dim].item(), sigma=sigma_list[action_dim].item())))) # normal dist
        # action = np.clip(random.lognormvariate(mu=mu.item(), sigma=sigma.item()), 256.0, 2048.0) # log dist
        if mode == "init":
            next_state, reward, over = env.step(action, mode)
            data.append((last_state, action, reward, next_state, over))
            reward_sum += reward
        elif mode == "train":
            next_state, reward, over = env.step(action, mode)
            for j in range(state_dim):
                last_state[j] = (last_state[j]-norm_mean[j]) / norm_std[j]
            data.append((last_state, action, reward, next_state, over))
            reward_sum += reward
        else:
            next_state, reward, over, frame_mse, frame_bitrate = env.step(action, mode)
            for j in range(state_dim):
                last_state[j] = (last_state[j]-norm_mean[j]) / norm_std[j]
            data.append((last_state, action, reward, next_state, over))
            reward_sum += reward
            video_mse += frame_mse
            video_bitrate += frame_bitrate
        
        last_state = next_state
        
    if mode == 'test':
        # print("Verifying: mean mse: ", video_mse/env.gop, "mean bitrate: ", video_bitrate/env.gop)
        return data, reward_sum, video_mse/env.gop, video_bitrate/env.gop
    
    return data, reward_sum


def get_action_entropy(model_action, state):
    mu, sigma = model_action(state)
    dist = torch.distributions.Normal(mu, sigma)
    action = dist.rsample()
    action = action.sigmoid()

    return action, sigma

class Pool:
    def __init__(self):
        self.pool = []
        self.mean = []
        self.std = []
    
    def __len__(self):
        return len(self.pool)
        
    def __getitem__(self, i):
        return self.pool[i]

    def get_norm_params(self):
        s = [[], [], [], [], [], [], [], [], [], [], [], []]
        for i in range(len(self.pool)):
            for j in range(12):
                for k in range(state_dim):
                    s[k].append(self.pool[i][j][0][k])
        for m in range(state_dim):
            self.mean.append(np.mean(s[m]))
            self.std.append(np.std(s[m]))
        print("===>")
        print("Init Pool State mean: ", self.mean)
        print("Init Pool State std: ", self.std)
        print("<===")

    def norm(self):
        for i in range(len(self.pool)):
            for j in range(12):
                for k in range(state_dim):
                    self.pool[i][j][0][k] = (self.pool[i][j][0][k] - self.mean[k]) / self.std[k]

    def init_fill(self, model_action, env):
        old_len = len(self.pool)
        while len(self.pool) - old_len < 160:
            self.pool.append(encode_one_video(model_action, env, mode='init', norm_mean=self.mean, norm_std=self.std)[0])  
        self.pool = self.pool[-160: ]
        self.get_norm_params()
        # self.norm()
    
    def update(self, model_action, env):
        # print("Updating the data pool...")
        old_len = len(self.pool)
        while len(self.pool) - old_len < 160:
            self.pool.append(encode_one_video(model_action, env, mode='train', norm_mean=self.mean, norm_std=self.std)[0])
        self.pool = self.pool[-160: ]
        # self.norm()
        # print("Updating done, current length of pool: ", len(self.pool))
        
    def push(self, size=32):
        state = []
        action = []
        reward = []
        next_state = []
        over = []
        if len(self.pool) <= size:
            for i in range(len(self.pool)):
                for j in range(12):
                    state.extend(self.pool[i][j][0])
                    action.append(self.pool[i][j][1])
                    reward.append(self.pool[i][j][2])
                    next_state.extend(self.pool[i][j][3])
                    over.append(self.pool[i][j][4])
            state = torch.tensor(state, dtype=torch.float32, device=device).reshape(-1, state_dim)
            action = torch.tensor(action, dtype=torch.float32, device=device).reshape(-1, 1)
            reward = torch.tensor(reward, dtype=torch.float32, device=device).reshape(-1, 1)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).reshape(-1, state_dim)
            over = torch.tensor(over, dtype=torch.float32, device=device).reshape(-1, 1)
            self.pool = []
            return state, action, reward, next_state, over
        
        data = self.pool[:size]
        for i in range(size):
            for j in range(12):
                state.extend(data[i][j][0])
                action.append(data[i][j][1])
                reward.append(data[i][j][2])
                next_state.extend(data[i][j][3])
                over.append(data[i][j][4])
        state = torch.tensor(state, dtype=torch.float32, device=device).reshape(-1, state_dim)
        action = torch.tensor(action, dtype=torch.float32, device=device).reshape(-1, 1)
        reward = torch.tensor(reward, dtype=torch.float32, device=device).reshape(-1, 1)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).reshape(-1, state_dim)
        over = torch.tensor(over, dtype=torch.float32, device=device).reshape(-1, 1)
        self.pool = self.pool[size:]
        return state, action, reward, next_state, over
        
    def sample(self):
        data = random.sample(self.pool, 32)
        state = []
        action = []
        reward = []
        next_state = []
        over = []
        # print(len(data), "sample_data: ", data)
        for i in range(64):
            for j in range(12):
                state.extend(data[i][j][0])
                action.append(data[i][j][1])
                reward.append(data[i][j][2])
                next_state.extend(data[i][j][3])
                over.append(data[i][j][4])
        state = torch.tensor(state, dtype=torch.float32, device=device).reshape(-1, state_dim)
        action = torch.tensor(action, dtype=torch.float32, device=device).reshape(-1, 1)
        reward = torch.tensor(reward, dtype=torch.float32, device=device).reshape(-1, 1)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).reshape(-1, state_dim)
        over = torch.tensor(over, dtype=torch.float32, device=device).reshape(-1, 1)
        
        return state, action, reward, next_state, over

def train_action(model_action, model_value1, model_value2, optimizer_action, state, entropy_weight):
    requires_grad(model_action, True)
    requires_grad(model_value1, False)
    requires_grad(model_value2, False)

    action, entropy = get_action_entropy(model_action, state)
    # action1, action2 = action.chunk(2, dim=0)
    _input = torch.cat([state, action], dim=1)

    value1 = model_value1(_input)
    value2 = model_value2(_input)
    value = torch.min(value1, value2)
    # value = torch.sum(prob * torch.min(value1, value2), dim=1, keepdims=True) # discrete TD3
    # print("train_action", value1, value2, entropy)
    # print("train_action")
    # print(value1, value2, entropy1, entropy2)
    # print("grad")
    # print(value1.grad, value2.grad, entropy1.grad, entropy2.grad)
    # print("train_action", value, entropy)
    loss = -(value+entropy_weight*(entropy)).mean()
    
    loss.backward()
    optimizer_action.step()
    # print("===grad===")
    # for name, parms in model_action.named_parameters():
    #     print('-->name:', name)
    #     # print('-->para:', parms)
    #     print('-->grad_requirs:', parms.requires_grad)
    #     if parms.grad != None:
    #         print('-->grad_value:', parms.grad.mean(), parms.grad.min(), parms.grad.max())
    #         print('-->grad_fn:', parms.grad_fn)
    #     else:
    #         print('-->grad_value: None')
    #     print("===")
    optimizer_action.zero_grad()

    return loss.item()
    
def train_value(model_action, model_action_delay, model_value1, model_value1_delay, model_value2, model_value2_delay, optimizer_value1, optimizer_value2, state, action, reward, next_state, over, entropy_weight):
    requires_grad(model_action, False)
    requires_grad(model_value1, True)
    requires_grad(model_value2, True)

    action1, action2 = action.chunk(2, dim=0)
    _input = torch.cat([state, action1, action2], dim=1)
    value1 = model_value1(_input)
    value2 = model_value2(_input)
    
    with torch.no_grad():
        next_action, entropy = get_action_entropy(model_action, next_state)  # SAC
        # next_action1, next_action2 = next_action.chunk(2, dim=0)
        # print(next_state.shape, next_action1.shape, next_action2.shape, next_action.shape)
        next_input = torch.cat([next_state, next_action], dim=1)
        target1 = model_value1_delay(next_input)
        target2 = model_value2_delay(next_input)
        target = torch.min(target1, target2)
    # target = torch.sum(prob * target, dim=1, keepdims=True) # discrete TD3
    target = target + entropy_weight * entropy
    target = target * 0.98 * (1-over) + reward # decay factor = 0.98
    
    loss1 = F.mse_loss(value1, target)
    loss2 = F.mse_loss(value2, target)
    
    loss1.backward()
    optimizer_value1.step()
    # print("===grad===")
    # for name, parms in model_value1.named_parameters():
    #     print('-->name:', name)
    #     # print('-->para:', parms)
    #     print('-->grad_requirs:', parms.requires_grad)
    #     if parms.grad != None:
    #         print('-->grad_value:', parms.grad.mean(), parms.grad.min(), parms.grad.max())
    #         print('-->grad_fn:', parms.grad_fn)
    #     else:
    #         print('-->grad_value: None')
    #     print("===")
    optimizer_value1.zero_grad()
    
    loss2.backward()
    optimizer_value2.step()
    # print("===grad===")
    # for name, parms in model_value1.named_parameters():
    #     print('-->name:', name)
    #     # print('-->para:', parms)
    #     print('-->grad_requirs:', parms.requires_grad)
    #     if parms.grad != None:
    #         print('-->grad_value:', parms.grad.mean(), parms.grad.min(), parms.grad.max())
    #         print('-->grad_fn:', parms.grad_fn)
    #     else:
    #         print('-->grad_value: None')
    #     print("===")
    optimizer_value2.zero_grad()
    
    return loss1.item(), loss2.item()

def train(model_action, model_value1, model_value2, model_action_delay, model_value1_delay, model_value2_delay, optimizer_action, optimizer_value1, optimizer_value2, env):
    if not os.path.exists("dummy_path_for_paper/experiment_name/run/"):
        os.makedirs("dummy_path_for_paper/experiment_name/run/", exist_ok=True)
    writer = SummaryWriter(log_dir="dummy_path_for_paper/experiment_name/run/")

    model_action.train()
    model_value1.train()
    model_value2.train()
    pool = Pool()
    pool.init_fill(model_action, env)
    pool.pool = []
    train_step = 0

    for epoch in tqdm(range(500), ncols=100): # in fact 2 videos each epoch, but prepared video nums is much larger to simulate online video streaming
        print("===Epoch===:", epoch)
        entropy_weight = 5e-4 - (5e-4 - 5e-6) * epoch / 500
        pool.update(model_action, env)
        # print(pool.pool)
        sum_reward = 0.0
        print("===>")
        print("Start Training")
        print("<===")
        while len(pool) > 0:
            state, action, reward, next_state, over = pool.push(size=32)
            
            value1_loss, value2_loss = train_value(model_action, model_action_delay, model_value1, model_value1_delay, model_value2, model_value2_delay, optimizer_value1, optimizer_value1, state, action, reward, next_state, over, entropy_weight)
            sum_reward += sum(reward)
            if train_step % 2 == 0:
                action_loss = train_action(model_action, model_value1, model_value2, optimizer_action, state, entropy_weight)
                writer.add_scalars("Training for actor(Policy Delay=2)", {"action_loss": action_loss, "value1_loss": value1_loss, "value2_loss": value2_loss}, train_step)
                soft_update(model_action, model_action_delay)
                soft_update(model_value1, model_value1_delay)
                soft_update(model_value2, model_value2_delay)
            train_step += 1
        writer.add_scalar("Training", sum_reward, epoch)
        print("===>")
        print("Training sum reward", sum_reward, epoch)
        print("<===")
        
        print("===>")
        print("Start Verifying")
        print("<===")
        sum_test_sum_reward = sum_test_video_mse = sum_test_video_bpp = 0.0
        cur_train_video = env.current_video
        env.current_video = 0
        while env.current_video < len(env.test_sequences) - 1:
            _, test_video_sum_reward, test_video_mse, test_video_bpp = encode_one_video(model_action, env, mode='test', norm_mean=pool.mean, norm_std=pool.std)
            sum_test_sum_reward += test_video_sum_reward
            sum_test_video_mse += test_video_mse
            sum_test_video_bpp += test_video_bpp
        env.current_video = cur_train_video
        mean_test_sum_reward = sum_test_sum_reward / len(env.test_sequences)
        mean_test_video_mse = sum_test_video_mse / len(env.test_sequences)
        mean_test_video_bpp = sum_test_video_bpp / len(env.test_sequences)
        writer.add_scalar("Verifying", mean_test_sum_reward, epoch)
        writer.add_scalar("Verifying", mean_test_video_mse, epoch)
        writer.add_scalar("Verifying", mean_test_video_bpp, epoch)
        print("===>")
        print("Verifying mean sum reward", sum_test_sum_reward / len(env.test_sequences), epoch)
        print("Verifying mean mse", sum_test_video_mse / len(env.test_sequences), epoch)
        print("Verifying mean bpp", sum_test_video_bpp / len(env.test_sequences), epoch)
        print("<===")
        writer.flush()
        os.makedirs("dummy_path_for_paper/ckpts/experiment_name/actor/", exist_ok=True)
        os.makedirs("dummy_path_for_paper/ckpts/experiment_name/actor_delay/", exist_ok=True)
        os.makedirs("dummy_path_for_paper/ckpts/experiment_name/critic1/", exist_ok=True)
        os.makedirs("dummy_path_for_paper/ckpts/experiment_name/critic1_delay/", exist_ok=True)
        os.makedirs("dummy_path_for_paper/ckpts/experiment_name/critic2/", exist_ok=True)
        os.makedirs("dummy_path_for_paper/ckpts/experiment_name/critic2_delay/", exist_ok=True)
        actor_checkpoint_path = ("dummy_path_for_paper/ckpts/experiment_name/actor/epoch_{}_loss_{:.7f}.pkl".format(epoch, action_loss))
        torch.save(model_action.state_dict(), actor_checkpoint_path)
        critic1_checkpoint_path = ("dummy_path_for_paper/ckpts/experiment_name/critic1/epoch_{}_loss_{:.7f}.pkl".format(epoch, value1_loss))
        torch.save(model_value1.state_dict(), critic1_checkpoint_path)
        critic2_checkpoint_path = ("dummy_path_for_paper/ckpts/experiment_name/critic2/epoch_{}_loss_{:.7f}.pkl".format(epoch, value2_loss))
        torch.save(model_value2.state_dict(), critic2_checkpoint_path)

        actor_delay_checkpoint_path = ("dummy_path_for_paper/ckpts/experiment_name/actor_delay/epoch_{}_loss_{:.7f}.pkl".format(epoch, action_loss))
        torch.save(model_action_delay.state_dict(), actor_delay_checkpoint_path)
        critic1_delay_checkpoint_path = ("dummy_path_for_paper/ckpts/experiment_name/critic1_delay/epoch_{}_loss_{:.7f}.pkl".format(epoch, value1_loss))
        torch.save(model_value1_delay.state_dict(), critic1_delay_checkpoint_path)
        critic2_delay_checkpoint_path = ("dummy_path_for_paper/ckpts/experiment_name/critic2_delay/epoch_{}_loss_{:.7f}.pkl".format(epoch, value2_loss))
        torch.save(model_value2_delay.state_dict(), critic2_delay_checkpoint_path)
    writer.close()


def main():
    global state_dim, action_dim, experiment_name
    state_dim = 10
    action_dim = 2
    experiment_name = "rdo_5e-4_rd-50remainr"
    print("Initing env...")
    env = Encoder_and_Decoder(experiment_name=experiment_name)
    print("Initing models...")
    model_action = Actor(state_dim=state_dim, action_dim=action_dim).to(device)
    model_action_delay = Actor(state_dim=state_dim, action_dim=action_dim).to(device)
    model_value1 = Value(state_plus_action_dim=state_dim+action_dim).to(device)
    model_value1_delay = Value(state_plus_action_dim=state_dim+action_dim).to(device)
    model_value2 = Value(state_plus_action_dim=state_dim+action_dim).to(device)
    model_value2_delay = Value(state_plus_action_dim=state_dim+action_dim).to(device)

    print("Initing optimizer...")
    optimizer_action = torch.optim.Adam(model_action.parameters(), lr=5e-4)
    optimizer_value1 = torch.optim.Adam(model_value1.parameters(), lr=5e-3)
    optimizer_value2 = torch.optim.Adam(model_value2.parameters(), lr=5e-3)

    print("Training...")
    train(model_action, model_value1, model_value2, model_action_delay, model_value1_delay, model_value2_delay, optimizer_action, optimizer_value1, optimizer_value2, env)  
             
if __name__ == "__main__":
    main()