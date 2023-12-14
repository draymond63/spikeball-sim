import numpy as np
import pybullet as p
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
import sys

from bullet import SpikeBallSimulator
from shot_optimize import optimize_pocket_shot


def optimize_net(*args, samples_per_type=5, **kwargs):
    df = get_data(*args, **kwargs)
    df = df.groupby(['type_in', 'type_out']).head(samples_per_type)
    print(df)
    input_states = df[['rim_dist_in', 'vx_in', 'vy_in']].to_numpy()
    output_states = df[['rim_dist_out', 'vx_out', 'vy_out']].to_numpy()
    return optimize_net_sim(input_states, output_states)

def optimize_net_sim(input_states: np.ndarray, output_states: np.ndarray):
    """Finds the optimal net parameters to match the output states given the input states"""
    assert input_states.shape == output_states.shape
    assert len(input_states.shape) == 2

    current_mass, current_scale = 0.1, 1
    sim = SpikeBallSimulator(current_mass, current_scale, plot=False)
    status = "Starting optimization..."

    pbar = tqdm()
    def objective(x):
        mass, net_scale = x
        nonlocal current_scale, current_mass, sim, status
        pbar.set_description(status + " GEN ")
        if mass != current_mass or net_scale != current_scale:
            p.disconnect()
            sim = SpikeBallSimulator(mass, net_scale, plot=False)
            current_mass, current_scale = mass, net_scale
        pbar.set_description(status + " SIM ")
        try:
            sim_out = np.array([sim.get_output_state(*input_state) for input_state in input_states])
        except Exception as e:
            raise Exception(f"Error in simulation with mu={mass}, lambda={x[1]}") from e
        error = sim_error(output_states, sim_out)
        status = f"mass={mass:.3f}, scale={net_scale:.3f} -> error={error:.3f}"
        pbar.set_description(status + " FIN ")
        pbar.update()
        return error

    x0 = np.array([current_mass, current_scale])
    res = minimize(objective, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    pbar.close()
    print(res)
    return res.x

def sim_error(correct_out: np.ndarray, sim_out: np.ndarray) -> float:
    """Returns the error between the correct output and the simulated output"""
    assert len(correct_out.shape) <= 2
    assert correct_out.shape == sim_out.shape
    return np.sum(np.linalg.norm(correct_out - sim_out, axis=-1))/len(correct_out)


def get_data(path='ball_tracking.csv'):
    df = pd.read_csv(path)
    vx_sign = np.sign(df['vx_in'])
    df['vx_in'] *= vx_sign # Pretend all shots are traveling rightward
    df['vx_out'] *= vx_sign
    df['rim_dist_in'] = np.abs(df['rim_dist_in'])/100
    df['rim_dist_out'] = np.abs(df['rim_dist_out'])/100
    return df

def get_shot_types(df):
    df = df.filter(['type_in', 'type_out'])
    df = df.groupby(['type_in', 'type_out']).size().reset_index(name='counts')
    df = df.pivot(index='type_in', columns='type_out', values='counts')
    df = df.drop('vertical', axis=1)
    df = df.div(df.sum(axis=1), axis=0)
    df = df.fillna(0)
    df = df.reindex(['low', 'medium', 'high'])
    df = df.round(2)
    return df

def get_pockets_by_shot(df):
    df = df.filter(['type_in', 'vx_in', 'vy_in', 'type_out'])
    # Drop vertical shots
    df = df[df['type_out'] != 'vertical']
    df['speed'] = np.linalg.norm(df[['vx_in', 'vy_in']], axis=1)
    median_speed = df['speed'].median()
    df['speed'] = df['speed'].apply(lambda x: 'slow' if x < median_speed else 'fast')
    # The final result should be a table where the columns are the speed types, the rows are the shot types, and the values are the percentage of pocket shots
    df = df.groupby(['type_in', 'speed', 'type_out']).size().reset_index(name='counts')
    df = df.pivot(index=['type_in', 'speed'], columns='type_out', values='counts')
    df = df.fillna(0)
    df = df.div(df.sum(axis=1), axis=0)
    df = df.round(2)
    df = df.reset_index()
    df.drop('normal', axis=1, inplace=True)
    df = df.pivot(index='type_in', columns='speed', values='pocket')
    return df


if __name__ == "__main__":
    with open('full.log', 'w+') as f:
        sys.stdout = f
        sys.stderr = f
        net_mass, net_scale = optimize_net()
        print(f"\nOptimal net parameters: mass={net_mass}, scale={net_scale}")
        try:
            print("\nOptimizing air time...")
            optimize_pocket_shot(mass=net_mass, scale=net_scale, opt_func='min_air_time')
        except Exception as e:
            print(e)
        try:
            print("\nOptimizing X-dist...")
            optimize_pocket_shot(mass=net_mass, scale=net_scale, opt_func='max_xdist')
        except Exception as e:
            print(e)
        try:
            print("\nOptimizing angle...")
            optimize_pocket_shot(mass=net_mass, scale=net_scale, opt_func='min_angle')
        except Exception as e:
            print(e)
