import json
import numpy as np
import os
import pickle

def get_trajectories(dirs):
    '''
    dirs: list of directories containing trajectories.pkl

    Returns list of np arrays, where each array is (N, 3)
    '''
    trajectories = []
    for dir in dirs:
        with open(os.path.join(dir, 'trajectories.pkl'), 'rb') as f:
            data = pickle.load(f)
            for traj in data:
                traj = np.array(traj)
                trajectories.append(traj)
    return trajectories

def normalize_trajectory(trajectory):
    '''
    Translates trajectory so that it start at [0,0] and rotates so that it ends on y-coordinate 0

    trajectory: (N, 3)
    '''
    # trajectory: (N, 3)
    traj = trajectory.copy()

    # Make traj start at [0,0]
    traj[:, 0] = traj[:, 0] - traj[0, 0]
    traj[:, 1] = traj[:, 1] - traj[0, 1]

    # Rotate traj about [0,0] so that it ends on y-coordinate 0
    last_point = traj[-1]
    angle = -np.arctan2(last_point[1], last_point[0])
    new_traj = np.zeros_like(traj)
    new_traj[:, 0] = traj[:, 0] * np.cos(angle) - traj[:, 1] * np.sin(angle)
    new_traj[:, 1] = traj[:, 0] * np.sin(angle) + traj[:, 1] * np.cos(angle)
    new_traj[:, 2] = traj[:, 2]
    traj = new_traj

    return traj

def resample_trajectory(trajectory, num_points=32):
    '''
    Resamples trajectory so that the number of points is num_points.
    Interpolates between points of original trajectory.

    trajectory: (N, 3)
    num_points: int
    '''
    N = trajectory.shape[0]
    start_times = [0]
    for i in range(1, N):
        start_times.append(start_times[-1] + np.linalg.norm(trajectory[i] - trajectory[i-1]))
    start_times = np.array(start_times)
    start_times = start_times / start_times[-1]

    res = []
    for i in range(num_points):
        t = i / (num_points - 1)

        # Get which points this is between
        idx_after = np.searchsorted(start_times, t, side='right')
        if idx_after == N:
            idx_after = N - 1
        idx_before = idx_after - 1
        assert(idx_after > 0)

        # Interpolate
        t_after = start_times[idx_after]
        t_before = start_times[idx_before]
        p_after = trajectory[idx_after]
        p_before = trajectory[idx_before]
        ratio = (t - t_before) / (t_after - t_before)
        res.append(p_before * (1 - ratio) + p_after * ratio)
    return np.array(res)

def plot_trajectory(ax, trajectory):
    # Draw line through trajectory points
    ax.plot(trajectory[:, 0], trajectory[:, 1], c='black', linewidth=0.5)

    # Draw small dots at trajectory points based on value of z-coordinate
    ax.scatter(trajectory[:, 0], trajectory[:, 1], c=trajectory[:, 2], s=1)