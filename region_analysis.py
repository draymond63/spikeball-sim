import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

from bullet import SpikeBallSimulator
from shot_optimize import rebound_dist_from_rim, RIM_DIST_BOUNDS
from net_fit import get_data



def sweep_until_pocket(sim: SpikeBallSimulator, x_start, x_end, vx, vy, incr=0.05):
    # Returns the distance from the rim that the ball will bounce off of
    iterations = 1
    rim_dist = x_start
    direction = -1 if x_start > x_end else 1
    incr *= direction
    while (x_end - rim_dist)*direction >= 0:
        xdist_out, vx_out, vy_out = sim.get_output_state(rim_dist, vx, vy)
        if vx_out < 0 and vy_out < 0 and xdist_out > 0 and rebound_dist_from_rim(xdist_out, vx_out, vy_out) > 0:
            # print(f"Found pocket region in {iterations} iterations")
            return rim_dist
        else:
            rim_dist += incr
            iterations += 1

# def precision_sweep_until_pocket(sim: SpikeBallSimulator, x_start, x_end, vx, vy, incr=0.05, sweep_scale=4):
#     rim_dist = sweep_until_pocket(sim, x_start, x_end, vx, vy, incr*sweep_scale)
#     if rim_dist is None:
#         return None
#     rim_dist = sweep_until_pocket(sim, rim_dist - incr*sweep_scale, rim_dist + incr*sweep_scale, vx, vy, incr)
#     if rim_dist is None:
#         raise Exception("Failed to find pocket region, after previously finding one")
#     return rim_dist

def get_pocket_region(sim: SpikeBallSimulator, vx, vy, incr=0.005):
    speed = np.linalg.norm([vx, vy])
    if rebound_dist_from_rim(RIM_DIST_BOUNDS[0], -speed/np.sqrt(2), -speed/np.sqrt(2)) < 0:
        return None
    # Returns the region of the rim that the ball will bounce off of
    region_start = sweep_until_pocket(sim, RIM_DIST_BOUNDS[0], RIM_DIST_BOUNDS[1], vx, vy, incr)
    if region_start is None:
        return None
    region_end = sweep_until_pocket(sim, RIM_DIST_BOUNDS[1], region_start, vx, vy, incr)
    max_iter = 10
    while region_end is None and max_iter > 0:
        region_end = sweep_until_pocket(sim, region_start + incr, region_start, vx, vy, incr)
        incr /= 2
        max_iter -= 1
    return (region_start, region_end)


def get_pocket_heatmap(speed_range, angle_range, count=3):
    sim = SpikeBallSimulator(mass=0.108, scale=0.925, max_duration=0.05)
    # Sweep through all possible input states
    angles, speeds = np.meshgrid(np.linspace(*angle_range, count), np.linspace(*speed_range, count))
    angles = angles.flatten()
    speeds = speeds.flatten()
    pocket_regions = []
    pbar = tqdm(total=len(angles))
    for angle, speed in zip(angles, speeds):
        vx = speed*np.cos(np.deg2rad(angle))
        vy = speed*np.sin(np.deg2rad(angle))
        pocket_region = get_pocket_region(sim, vx, vy)
        pbar.update()
        pbar.set_description(f"vx={vx:.3f}, vy={vy:.3f} -> {pocket_region}")
        if pocket_region is None:
            pocket_region = (0, 0)
        pocket_regions.append(pocket_region)
    # Plot heatmap with speed and shot angle as the axes, and region size as the color
    pocket_regions = np.array(pocket_regions)
    df = pd.DataFrame({'speed': speeds, 'angle': angles, 'start': pocket_regions[:, 0], 'end': pocket_regions[:, 1]})
    return df


def plot_heatmap(df: pd.DataFrame, pivoted=False):
    fig = go.Figure(data=go.Heatmap(
        x=df.columns,
        y=df.index,
        z=df.values,
        text=df.values,
        colorscale='Reds',
    )) if pivoted else (
    go.Figure(data=go.Heatmap(
        x=df['speed'],
        y=df['angle'],
        z=df['end'] - df['start'],
        text=df['end'] - df['start'],
        colorscale='Inferno',
    )))
    fig.update_layout(
        title='Pocket Region Size',
        xaxis_title='Speed (m/s)',
        yaxis_title='Angle (deg)',
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=10,
        ),
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    fig.show()




def pivot_pocket_map(df: pd.DataFrame):
    df['size'] = df['end'] - df['start']
    df = pd.pivot_table(df, values='size', index=['speed'], columns=['angle'])
    df = df.fillna(0)
    df = df.sort_index(ascending=True)
    return df

def melt_pocket_map(df: pd.DataFrame):
    df = pd.melt(df, id_vars=['speed'], var_name='angle', value_name='size')
    df = df.sort_values(by=['speed', 'angle'], ascending=False)
    df['angle'] = df['angle'].astype(float)
    return df

def increase_map_fidelity(df: pd.DataFrame) -> pd.DataFrame:
    # Assume columns are speed, angle, start, end
    speeds = sorted(df['speed'].unique())
    angles = sorted(df['angle'].unique())
    speed_incr = np.abs(speeds[1] - speeds[0])/2
    angle_incr = np.abs(angles[1] - angles[0])/2
    min_df = df[(df['end'] - df['start']) != 0]
    pairs = set()
    for speed, angle in zip(min_df['speed'], min_df['angle']):
        pairs.add((speed, angle - angle_incr))
        pairs.add((speed, angle + angle_incr))
        pairs.add((speed - speed_incr, angle))
        pairs.add((speed + speed_incr, angle))
        pairs.add((speed - speed_incr, angle - angle_incr))
        pairs.add((speed - speed_incr, angle + angle_incr))
        pairs.add((speed + speed_incr, angle - angle_incr))
        pairs.add((speed + speed_incr, angle + angle_incr))
    # Get new data
    sim = SpikeBallSimulator(mass=0.108, scale=0.925, max_duration=0.05, plot=False)
    new_data = []
    for speed, angle in tqdm(pairs):
        vx = speed*np.cos(np.deg2rad(angle))
        vy = speed*np.sin(np.deg2rad(angle))
        pocket_region = get_pocket_region(sim, vx, vy)
        if pocket_region is None:
            pocket_region = (0, 0)
        new_data.append((speed, angle, *pocket_region))
    df2 = pd.DataFrame(new_data, columns=df.columns)
    df = pd.concat([df, df2])
    df = df.sort_values(by=['speed', 'angle'], ascending=False)
    df = df.drop_duplicates(subset=['speed', 'angle'], keep='first')
    df = df.reset_index(drop=True)
    return df

def smooth_regions(df):
    df = pivot_pocket_map(df).transpose()
    df = df.interpolate(method='linear', axis=1)
    speed_spacing = np.abs(df.columns[1] - df.columns[0])
    df = df.reindex(columns=df.columns.union(np.arange(df.columns[0] - 2*speed_spacing, df.columns[0], speed_spacing)), fill_value=0)
    # Smooth data
    df = df.transform(lambda x: x.rolling(2, center=True, min_periods=1).mean(), axis=1)
    df = df.transform(lambda x: x.rolling(2, center=True, min_periods=1).mean(), axis=0)
    return df


if __name__ == "__main__":
    speed_range = (10, 25)
    angle_range = (40, 90)
    df = get_pocket_heatmap(speed_range, angle_range, count=10)
    df.to_csv('pocket_heatmap.csv', index=False)
    plot_heatmap(df)

    # df = pd.read_csv('pocket_heatmap.csv')
    df = increase_map_fidelity(df)
    df.to_csv('pocket_heatmap-2.csv', index=False)
    plot_heatmap(df)

    # df = pd.read_csv('pocket_heatmap-2.csv')
    df = smooth_regions(df)
    plot_heatmap(df, pivoted=True)