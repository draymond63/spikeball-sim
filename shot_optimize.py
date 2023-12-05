import numpy as np
from scipy.optimize import differential_evolution, LinearConstraint, NonlinearConstraint
from tqdm import tqdm
import sys

from bullet import GRAVITY, BALL_RADIUS, NET_RADIUS, SpikeBallSimulator


def compute_angle(vx_out, vy_out):
    rad = np.arctan2(-vy_out, -vx_out)
    if rad < 0:
        rad += 2*np.pi
    return np.rad2deg(rad)

def get_air_time(vy_out, g=GRAVITY):
    return -vy_out/g

def rebound_dist_from_rim(rim_dist, vx, vy, g=GRAVITY, raise_on_invalid=True):
    # Measured from closest point on rim, positive is away
    if vy >= 0:
        if raise_on_invalid:
            raise ValueError("vy must be negative (upward))")
        else:
            return np.inf
    if vx >= 0:
        if raise_on_invalid:
            raise ValueError("vx must be negative (rightward)")
        else:
            return np.inf
    x = - 2*NET_RADIUS + rim_dist
    t = get_air_time(vy, g)
    return x - vx*t - 2*NET_RADIUS


def optimize_pocket_shot(mass: float, scale: float, max_v=22.81563379224826, opt_func='max_xdist'):
    # Find the optimal pocket shot, i.e. the shot rebounds at the shallowest angle possible while still clearing the net
    sim = SpikeBallSimulator(mass, scale, plot=False)
    pbar = tqdm()

    def max_xdist(rim_dist, vx_out, vy_out):
        return vx_out # Want it to bounce as fast as possible in the negative x direction

    def min_angle(rim_dist, vx_out, vy_out):
        return compute_angle(vx_out, vy_out)
    
    def min_air_time(rim_dist, vx_out, vy_out):
        air_time = get_air_time(vy_out)
        if air_time < 0: # TODO: Necessary?
            return np.inf
        return air_time

    def max_tolerance(rim_dist, vx_out, vy_out):
        # Minimize varability of the output state given the input state by comput
        raise NotImplementedError()

    func_map = {
        'max_xdist': max_xdist,
        'min_angle': min_angle,
        'min_air_time': min_air_time,
        'max_tolerance': max_tolerance,
    }

    def objective(x):
        rim_dist, vx, vy = x
        xdist_out, vx_out, vy_out = sim.get_output_state(rim_dist, vx, vy)
        obj = func_map[opt_func](xdist_out, vx_out, vy_out)
        pbar.update()
        pbar.set_description(f"x={rim_dist:.5f}, vx={vx:.5f}, vy={vy:.5f} -> vx_out={vx_out:.3f}, vy_out={vy_out:.3f} (obj={obj:.3f})")
        return obj

    # rebound_constr = NonlinearConstraint(lambda x: sim.get_output_state(*x)[1], -np.inf, 0)
    # min_outbound_constr = NonlinearConstraint(lambda x: np.linalg.norm(sim.get_output_state(*x)[1:]), 0, np.inf)
    if opt_func == 'min_air_time':
        clear_rim_constr = NonlinearConstraint(lambda x: rebound_dist_from_rim(sim.get_output_state(*x), raise_on_invalid=False), 0, np.inf)
    rim_dist_bounds = [2*BALL_RADIUS, 2*NET_RADIUS - 2*BALL_RADIUS]
    vx_bounds = [0, max_v]
    vy_bounds = [0, max_v]
    vin_constr = LinearConstraint([0, 1/np.sqrt(2), 1/np.sqrt(2)], [0], [max_v], keep_feasible=True)
    res = differential_evolution(objective, bounds=[rim_dist_bounds, vx_bounds, vy_bounds], constraints=[vin_constr])
    pbar.close()
    print(res)
    return res.x


if __name__ == "__main__":
    optimize_pocket_shot(mass=0.108, scale=0.925, max_v=40, opt_func='min_angle')
    