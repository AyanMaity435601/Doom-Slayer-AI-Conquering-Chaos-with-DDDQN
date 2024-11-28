import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

import numpy as np
import random
import time
from vizdoom import *
from models import *

from collections import deque
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import imageio  # Import imageio for video creation
import os       # Import os to handle file paths

# Boolean specifying whether GPUs are available or not.
use_cuda = torch.cuda.is_available()

"""
Environment tools
"""

def create_environment(scenario='basic', window=False):
    

    game = DoomGame()
    if window:
        game.set_window_visible(True)
    else:
        game.set_window_visible(False)

    # Load the correct configuration
    if scenario == 'basic':
        game.load_config("scenarios/basic.cfg")
        game.set_doom_scenario_path("scenarios/basic.wad")
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]

    elif scenario == 'deadly_corridor':
        game.load_config("scenarios/deadly_corridor.cfg")
        game.set_doom_scenario_path("scenarios/deadly_corridor.wad")
        possible_actions = np.identity(6, dtype=int).tolist()

    elif scenario == 'defend_the_center':
        game.load_config("scenarios/defend_the_center.cfg")
        game.set_doom_scenario_path("scenarios/defend_the_center.wad")
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]

    game.set_screen_format(ScreenFormat.GRAY8)

    game.init()

    return game, possible_actions


def preprocess_frame_for_video(frame):
    
    if frame.ndim == 2:
        # Convert grayscale to RGB by duplicating the single channel
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        pass  # Frame is already RGB
    else:
        raise ValueError(f"Unexpected frame shape: {frame.shape}. Expected (height, width, 3) or (height, width).")

    # Ensure the frame is in uint8 format
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    return frame


def test_environment(weights, scenario='basic', window=False, total_episodes=100, enhance='none', frame_skip=2, stack_size=4, record=False, output_dir='.'):
    

    game = DoomGame()
    if window:
        game.set_window_visible(True)
    else:
        game.set_window_visible(False)

    # Load the correct configuration
    if scenario == 'basic':
        game.load_config("scenarios/basic.cfg")
        game.set_doom_scenario_path("scenarios/basic.wad")
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]

    elif scenario == 'deadly_corridor':
        game.load_config("scenarios/deadly_corridor.cfg")
        game.set_doom_scenario_path("scenarios/deadly_corridor.wad")
        possible_actions = np.identity(6, dtype=int).tolist()
        # possible_actions.extend([[0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1]])

    elif scenario == 'defend_the_center':
        game.load_config("scenarios/defend_the_center.cfg")
        game.set_doom_scenario_path("scenarios/defend_the_center.wad")
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]
        # possible_actions.extend([[1, 0, 1], [0, 1, 1]])

    # Set screen format to grayscale after loading config
    game.set_screen_format(ScreenFormat.GRAY8)

    game.init()

    if enhance == 'none':
        model = DQNetwork(stack_size=stack_size, out=len(possible_actions))
        if use_cuda:
            model.cuda()

    elif enhance == 'dueling':
        model = DDDQNetwork(stack_size=stack_size, out=len(possible_actions))
        if use_cuda:
            model.cuda()

    # Load the weights of the model
    state_dict = torch.load(weights, map_location=torch.device('cuda' if use_cuda else 'cpu'))
    model.load_state_dict(state_dict)
    model.eval()  # Set model to evaluation mode

    for episode in range(total_episodes):
        game.new_episode()
        if record:
            frames = []
        done = game.is_episode_finished()
        state = get_state(game)
        in_channels = model._modules['conv_1'].in_channels
        stacked_frames = deque([torch.zeros((120, 160), dtype=torch.float32) for _ in range(in_channels)], maxlen=in_channels)
        state, stacked_frames = stack_frames(stacked_frames, state, True, in_channels)
        while not done:
            # Capture frame if recording
            if record:
                state_game = game.get_state()
                if state_game:
                    frame = state_game.screen_buffer  # Grayscale frame
                    if frame is not None:
                        try:
                            frame_rgb = preprocess_frame_for_video(frame)
                            frames.append(frame_rgb)
                        except ValueError as ve:
                            print(f"Frame preprocessing failed: {ve}")
            if use_cuda:
                with torch.no_grad():
                    q = model(state.cuda())
            else:
                with torch.no_grad():
                    q = model(state)

            action = possible_actions[int(torch.max(q, 1)[1][0])]
            reward = game.make_action(action, frame_skip)
            done = game.is_episode_finished()
            if not done:
                state = get_state(game)
                state, stacked_frames = stack_frames(stacked_frames, state, False, in_channels)

            # Optional: Adjust sleep time or remove for faster execution
            time.sleep(0.02)

        print(f"Episode {episode + 1}/{total_episodes} - Total reward:", game.get_total_reward())
        # Optional: Adjust sleep time or remove for faster execution
        time.sleep(0.1)
        # Save the recorded frames as a video
        if record and frames:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            video_filename = os.path.join(output_dir, f'episode_{episode + 1}.mp4')
            try:
                imageio.mimsave(video_filename, frames, fps=30)
                print(f"Saved video: {video_filename}")
            except Exception as e:
                print(f"Failed to save video {video_filename}: {e}")

    game.close()


"""
Preprocessing tools
"""
def get_state(game):
    """
    Description
    --------------
    Get the current state from the game.

    Parameters
    --------------
    game : VizDoom game instance.

    Returns
    --------------
    state : numpy array, the current grayscale frame.
    """

    state = game.get_state().screen_buffer
    return state  # Return as-is since it's grayscale


def transforms_func(resize=(120, 160)):
    

    return T.Compose([
        T.ToPILImage(),
        T.Resize(resize),
        T.ToTensor()
    ])


def stack_frames(stacked_frames, state, is_new_episode, maxlen=4, resize=(120, 160)):
    

    # Preprocess frame
    frame = transforms_func(resize)(state)  # [1, H, W]
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([frame.squeeze(0) for _ in range(maxlen)], maxlen=maxlen)  # [H, W] each
        # Stack the frames
        stacked_state = torch.stack(tuple(stacked_frames), dim=0).unsqueeze(0)  # [1, 4, H, W]
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame.squeeze(0))  # [H, W]
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = torch.stack(tuple(stacked_frames), dim=0).unsqueeze(0)  # [1, 4, H, W]

    return stacked_state, stacked_frames


"""
epsilon-greedy
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, model, possible_actions):
    """
    Description
    -------------
    Epsilon-greedy policy

    Parameters
    -------------
    explore_start    : Float, the initial exploration probability.
    explore_stop     : Float, the last exploration probability.
    decay_rate       : Float, the rate at which the exploration probability decays.
    state            : 4D-tensor (batch, motion, image)
    model            : models.DQNetwork or models.DDDQNetwork object, the architecture used.
    possible_actions : List, the one-hot encoded possible actions.

    Returns
    -------------
    action              : np.array of shape (number_actions,), the action chosen by the greedy policy.
    explore_probability : Float, the exploration probability.
    """

    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if explore_probability > exp_exp_tradeoff:
        action = random.choice(possible_actions)

    else:
        if use_cuda:
            Qs = model.forward(state.cuda())
        else:
            Qs = model.forward(state)

        action = possible_actions[int(torch.max(Qs, 1)[1][0])]

    return action, explore_probability


"""
Double Q-learning tools
"""
def update_target(current_model, target_model):
    """
    Description
    -------------
    Update the parameters of target_model with those of current_model

    Parameters
    -------------
    current_model, target_model : torch models
    """
    target_model.load_state_dict(current_model.state_dict())


"""
Make gif
"""
def make_gif(images, fname, fps=50):
    import moviepy.editor as mpy  # Ensure moviepy is imported

    def make_frame(t):
        try:
            x = images[int(fps * t)]
        except:
            x = images[-1]
        return x.astype(np.uint8)
    
    clip = mpy.VideoClip(make_frame, duration=len(images) / fps)
    clip.fps = fps
    clip.write_gif(fname, program='ffmpeg', fuzz=50, verbose=False)
