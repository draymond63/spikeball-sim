import pybullet as p
import numpy as np
from scipy.optimize import minimize
from time import sleep

NET_RADIUS = 0.4572 # m
BALL_RADIUS = 0.04445 # m
NET_MODEL = "net.obj"
NET_EDGE_NODES = 520
# NET_NODES = 5720


class SpikeBallSimulator:
    def __init__(self, *net_params, max_duration=1, g=1, plot=False, trimetric=False, **net_kwargs) -> None:
        self.TIMESTEP = 240
        self.max_steps = int(max_duration*self.TIMESTEP)
        self.plot = plot
        p.connect(p.GUI if plot else p.DIRECT)
        # Camera settings
        if plot and not trimetric:
            cameraTargetPosition = [0, 0, 0]  # x, y, z
            cameraDistance = 2
            cameraYaw = np.pi/2
            cameraPitch = 0
            p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
        p.setGravity(0, 0, -g)
        # Net model
        self.contact_margin = 0.1
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

    def update_net(self, net_mu: float, net_lambda: float, net_friction: float = 0.5):
        if self.init_state is not None:
            p.restoreState(self.init_state)
        if self.net is not None:
            raise NotImplementedError("Updating net parameters is not yet implemented")
            p.removeBody(self.net)
        self.net = p.loadSoftBody(
            NET_MODEL,
            basePosition=[-0.5, 0.5, 0], # Center of the net is not at the origin
            baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0, 0]),
            scale=0.0005, # Model has a diameter of 1828.8. That is 6ft in mm, but we need to scale to meters
            mass=0.1,
            useNeoHookean=True,
            NeoHookeanMu=net_mu,
            NeoHookeanLambda=net_lambda,
            NeoHookeanDamping=0.01,
            useSelfCollision=1,
            frictionCoeff=net_friction,
            collisionMargin=self.contact_margin
        )
        # Fix rim
        for nodeIndex in range(NET_EDGE_NODES):
            p.createSoftBodyAnchor(self.net, nodeIndex, -1, -1)  # Anchor to a fixed point in space
        self.init_state = p.saveState()

    def run(self, rim_contact_dist, vx, vy, min_steps=4, let_run=False, raise_on_error=True):
        p.restoreState(self.init_state)
        p.resetBasePositionAndOrientation(self.ball, [NET_RADIUS - rim_contact_dist, 0, BALL_RADIUS], [0, 0, 0, 1])
        p.resetBaseVelocity(self.ball, linearVelocity=[vx, 0, -vy], angularVelocity=[0, 0, 0])

        ball_coords = []
        sim_started = False
        step_count = 0
        if self.plot:
            p.setRealTimeSimulation(True)
        while p.isConnected():
            if self.plot:
                sleep(0.01)  # Time in seconds.
                # p.getCameraImage(320,200,renderer=p.ER_BULLET_HARDWARE_OPENGL)
            else:
                p.stepSimulation()
            step_count += 1
            # Get ball position
            (x, y, z), _ = p.getBasePositionAndOrientation(self.ball)
            ball_coords.append((x, y, z))
            if not sim_started and z < BALL_RADIUS:
                sim_started = True
            elif not let_run and sim_started and z > BALL_RADIUS and step_count > min_steps:
                break
            elif step_count > self.max_steps:
                if let_run:
                    break
                raise Exception(f"Simulation timed out with input x={rim_contact_dist}, vx={vx}, vy={vy}")
            elif z < -3*BALL_RADIUS and raise_on_error:
                raise Exception(f"Ball fell through net with input x={rim_contact_dist}, vx={vx}, vy={vy}")
        return np.array(ball_coords) # (step_count, 3)

    def get_output_state(self, *input_state, **kwargs) -> tuple[float, float, float]:
        """Returns the output state (rim_dist, vx_out, vz_out) of the ball given the ball coordinates"""
        ball_coords = self.run(*input_state, **kwargs, let_run=False)
        rim_dist = NET_RADIUS - ball_coords[-1, 0]
        v_vec = p.getBaseVelocity(self.ball)[0]
        vx_out, vz_out = v_vec[0], -v_vec[2]
        return rim_dist, vx_out, vz_out



def sim_error(correct_out: np.ndarray, sim_out: np.ndarray) -> float:
    """Returns the error between the correct output and the simulated output"""
    assert len(correct_out.shape) <= 2
    assert correct_out.shape == sim_out.shape
    return np.sum(np.linalg.norm(correct_out - sim_out, axis=-1))


# Optimize material properties of the net to match the output state
def optimize_net_sim(input_states: np.ndarray, output_states: np.ndarray):
    assert len(input_states) == len(output_states)
    assert len(input_states.shape) == 2

    def objective(x):
        print(f"New iteration with mu={x[0]}, lambda={x[1]}")
        sim = SpikeBallSimulator(net_mu=x[0], net_lambda=x[1], plot=False)
        try:
            sim_out = np.array([sim.get_output_state(*input_state, raise_on_error=False) for input_state in input_states])
        except Exception as e:
            raise Exception(f"Error in simulation with mu={x[0]}, lambda={x[1]}") from e
        return sim_error(output_states, sim_out)

    init_mu = 1e3
    init_lambda = 5
    x0 = np.array([init_mu, init_lambda])
    res = minimize(objective, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    print(res)
    return res.x


if __name__ == "__main__":
    rim_dist_range = np.linspace(4*BALL_RADIUS, NET_RADIUS, 10)
    in_vx_range = np.linspace(0.1, 4, 10)
    in_vy_range = np.linspace(0.1, 4, 10)
    input_states = np.vstack([rim_dist_range, in_vx_range, in_vy_range]).T
    output_states = np.array([ # Simulated with net_mu=1050, net_lambda=10
        [0.17177305, -0.0119, -0.2394],
        [0.16058951, 0.1142, -0.84351],
        [0.15880679, 0.0138, -1.36432],
        [0.15881482, 0.1067, -1.82267],
        [0.14964238, 0.2735, -2.30739],
        [0.15231321, 0.2827, -2.79035],
        [0.17175946, 0.0828, -3.14165],
        [0.17279976, 0.2647, -3.62030],
        [0.17296509, 0.6257, -3.96120],
        [0.28664461, 0.0848, -3.71219],
    ])
    net_mu, net_lambda = optimize_net_sim(input_states, output_states)
    print(net_mu, net_lambda)

    # sim = SpikeBallSimulator(net_mu=1050, net_lambda=5, plot=True)
    # for state in input_states:
    #     print(state, end=" -> ")
    #     print(sim.get_output_state(*state))
