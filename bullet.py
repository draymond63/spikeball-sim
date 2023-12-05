import numpy as np
import pybullet as p
import pandas as pd
from PIL import Image
from scipy.optimize import minimize, LinearConstraint
from time import sleep
from tqdm import tqdm
import sys

np.seterr(invalid='raise')
NET_RADIUS = 0.42 # m
BALL_RADIUS = 0.04445 # m
NET_MODEL = "net.obj"
NET_EDGE_NODES = 520
# NET_NODES = 5720


class SpikeBallSimulator:
    def __init__(self, *net_params, max_duration=0.01, g=9.81, plot=False, trimetric=False, **net_kwargs) -> None:
        self.step_rate = 20000
        self.max_steps = int(max_duration*self.step_rate)
        self.plot = plot
        p.connect(p.GUI if plot else p.DIRECT)
        p.setTimeStep(1/self.step_rate)
        # Camera settings
        cameraTargetPosition = [0, 0, 0]  # x, y, z
        cameraDistance = 0.7
        cameraYaw = 0
        cameraPitch = -30 if trimetric else 0
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
        p.setGravity(0, 0, -g)
        # Net model
        self.contact_margin = 0.01
        # Ball model
        self.ball = p.createMultiBody(
            baseMass=0.150, # kg
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS),
            basePosition=[0, 0, BALL_RADIUS],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        )
        self.net = None
        self.init_state = None
        self.update_net(*net_params, **net_kwargs)
        self.init_state = p.saveState()

    def update_net(self, mass=0.1, scale=1):
        if self.init_state is not None:
            p.restoreState(self.init_state)
        if self.net is not None:
            raise NotImplementedError("Updating net parameters is not yet implemented")
            p.removeBody(self.net)
        self.net = p.loadSoftBody(
            NET_MODEL,
            basePosition=[-0.5, 0.5, 0], # Center of the net is not at the origin
            baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0, 0]),
            scale=0.0005*scale, # Model has a diameter of 1828.8. That is 6ft in mm, but we need to scale to meters
            mass=mass,
            useSelfCollision=1,
            collisionMargin=self.contact_margin,
        )
        # Fix rim
        for nodeIndex in range(NET_EDGE_NODES):
            p.createSoftBodyAnchor(self.net, nodeIndex, -1, -1)  # Anchor to a fixed point in space
        self.init_state = p.saveState()

    def run(self, rim_contact_dist, vx, vy, min_steps=4, save=None, demo=False):
        p.restoreState(self.init_state)
        contact_height = BALL_RADIUS + self.contact_margin/2
        # Start ball from higher up if it's a demo, but so that it contacts the net at the same point
        if demo:
            # p.setGravity(0, 0, 0)
            contact_height += vy/70
            rim_contact_dist += vx/70
        p.resetBasePositionAndOrientation(self.ball, [NET_RADIUS - rim_contact_dist, 0, contact_height], [0, 0, 0, 1])
        p.resetBaseVelocity(self.ball, linearVelocity=[vx, 0, -vy], angularVelocity=[0, 0, 0])

        ball_coords = []
        frames = []
        sim_started = False
        step_count = 0
        if self.plot:
            p.setRealTimeSimulation(True)
        while p.isConnected():
            if self.plot:
                sleep(0.01)  # Time in seconds.
            else:
                p.stepSimulation()
            if save is not None:
                width, height, rgbImg, depthImg, segImg = p.getCameraImage(720, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)
                rgb = np.array(rgbImg, dtype=np.uint8)
                rgb = np.reshape(rgb, (height, width, 4))[:, :, :3]
                frames.append(Image.fromarray(rgb, 'RGB'))
            step_count += 1
            # Get ball position
            (x, y, z), _ = p.getBasePositionAndOrientation(self.ball)
            ball_coords.append((x, y, z))
            if not sim_started and z < contact_height:
                sim_started = True
            elif not demo and sim_started and z > contact_height and step_count > min_steps:
                break
            elif np.linalg.norm([x, y]) > NET_RADIUS + BALL_RADIUS and not demo:
                break
            elif step_count > self.max_steps:
                if not sim_started:
                    raise Exception(f"Ball never hit net with input x={rim_contact_dist}, vx={vx}, vy={vy}")
                break
            elif z < -3*BALL_RADIUS and not demo:
                raise Exception(f"Ball fell through net with input x={rim_contact_dist}, vx={vx}, vy={vy}")
        if save:
            frames[0].save(save, format='GIF', append_images=frames[1:], save_all=True, duration=len(frames)/self.step_rate, loop=0)
        return np.array(ball_coords) # (step_count, 3)

    def get_output_state(self, *input_state, **kwargs) -> tuple[float, float, float]:
        """Returns the output state (rim_dist, vx_out, vy_out) of the ball given the ball coordinates"""
        ball_coords = self.run(*input_state, **kwargs, demo=False)
        rim_dist = NET_RADIUS - ball_coords[-1, 0]
        v_vec = p.getBaseVelocity(self.ball)[0]
        vx_out, vy_out = v_vec[0], -v_vec[2]
        return rim_dist, vx_out, vy_out



def sim_error(correct_out: np.ndarray, sim_out: np.ndarray) -> float:
    """Returns the error between the correct output and the simulated output"""
    assert len(correct_out.shape) <= 2
    assert correct_out.shape == sim_out.shape
    return np.sum(np.linalg.norm(correct_out - sim_out, axis=-1))/len(correct_out)



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


def optimize_pocket_shot(net_mass: float, net_scale: float, max_v=15, opt_func='max_xdist'):
    # Find the optimal pocket shot, i.e. the shot rebounds at the shallowest angle possible while still clearing the net
    sim = SpikeBallSimulator(net_mass, net_scale, plot=False)
    pbar = tqdm()

    def max_xdist(x, vx_out, vy_out):
        return vx_out # Want it to bounce as fast as possible in the negative x direction
    
    def min_angle(x, vx_out, vy_out):
        rad = np.arctan2(-vy_out, -vx_out)
        if rad < 0:
            rad += 2*np.pi
        return rad
    
    func_map = {
        'max_xdist': max_xdist,
        'min_angle': min_angle,
    }

    def objective(x):
        rim_dist, vx, vy = x
        xdist_out, vx_out, vy_out = sim.get_output_state(rim_dist, vx, vy)
        pbar.update()
        pbar.set_description(f"x={rim_dist:.5f}, vx={vx:.5f}, vy={vy:.5f} -> vx_out={vx_out:.3f}, vy_out={vy_out:.3f}")
        return func_map[opt_func](xdist_out, vx_out, vy_out)

    # clear_rim_constr = {
    #     'type': 'ineq',
    #     'fun': lambda x: NET_RADIUS - x[0]
    # }
    rim_dist_constr = LinearConstraint([1, 0, 0], [2*BALL_RADIUS], [2*NET_RADIUS - 2*BALL_RADIUS], keep_feasible=True)
    vxin_constr = LinearConstraint([0, 1, 0], [0], [max_v], keep_feasible=True) # Shot must be to the right
    vyin_constr = LinearConstraint([0, 0, 1], [0], [max_v], keep_feasible=True) # Shot must be downward
    vin_constr = LinearConstraint([0, 1/np.sqrt(2), 1/np.sqrt(2)], [0], [max_v], keep_feasible=True)
    res = minimize(objective, [3*BALL_RADIUS, 3, 3], method='trust-constr', constraints=[rim_dist_constr, vxin_constr, vyin_constr, vin_constr], options={'disp': True})
    pbar.close()
    print(res)
    return res.x


def get_data(path='ball_tracking.csv'):
    df = pd.read_csv(path)
    df['vx_in'] = np.abs(df['vx_in'])
    df['vx_in'] = np.abs(df['vx_in'])
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
    # vx_in,vy_in,vx_out,vy_out,rim_dist_in,rim_dist_out,angle_in,angle_out,type_in,type_out,path
    df = get_data()
    # Get one of each type of shot (i.e. in=low, out=pocket)
    # df = df.groupby(['type_in', 'type_out']).head(2) # Drops time from 97s/it to 14s/it
    # input_states = df[['rim_dist_in', 'vx_in', 'vy_in']].to_numpy()
    # output_states = df[['rim_dist_out', 'vx_out', 'vy_out']].to_numpy()
    # with open('log.txt', 'w+') as f:
    #     sys.stdout = f
    #     sys.stderr = f
    #     net_mass, net_scale = optimize_net_sim(input_states, output_states)
    # print(net_mass, net_scale)

    # state = input_states[34]
    # print(df.head())
    # print(state)
    # sim = SpikeBallSimulator(plot=True, trimetric=True)
    # sim.run(*state, save='test.gif', demo=True)

    # print(sim.get_output_state(*state))
    # for state in input_states:
    #     print(state, end=" -> ")
    #     print(sim.get_output_state(*state))

    # optimize_pocket_shot(mass=1050, net_scale=5, opt_func='max_xdist')
