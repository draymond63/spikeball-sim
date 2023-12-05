import numpy as np
import pybullet as p
from functools import cache
from PIL import Image
from time import sleep

np.seterr(invalid='raise')
NET_RADIUS = 0.42 # m
BALL_RADIUS = 0.04445 # m
NET_MODEL = "net.obj"
NET_EDGE_NODES = 520
GRAVITY = 9.81 # m/s^2
# NET_NODES = 5720


class SpikeBallSimulator:
    def __init__(self, *net_params, max_duration=0.1, g=GRAVITY, plot=False, trimetric=False, **net_kwargs) -> None:
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
                    raise Exception(f"Ball never hit net with input x={rim_contact_dist}, {vx}, vy={vy}")
                if not demo:
                    print("\nWarning: simulation timed out")
                break
            elif z < -3*BALL_RADIUS and not demo:
                raise Exception(f"Ball fell through net with input x={rim_contact_dist}, vx={vx}, vy={vy}")
        if save:
            frames[0].save(save, format='GIF', append_images=frames[1:], save_all=True, duration=len(frames)/self.step_rate, loop=0)
        return np.array(ball_coords) # (step_count, 3)

    @cache
    def get_output_state(self, *input_state, **kwargs) -> tuple[float, float, float]:
        """Returns the output state (rim_dist, vx_out, vy_out) of the ball given the ball coordinates"""
        ball_coords = self.run(*input_state, **kwargs)
        rim_dist = NET_RADIUS - ball_coords[-1, 0]
        v_vec = p.getBaseVelocity(self.ball)[0]
        vx_out, vy_out = v_vec[0], -v_vec[2]
        return rim_dist, vx_out, vy_out



if __name__ == "__main__":
    state = [0.13716, 3.72548, 3.57745]
    sim = SpikeBallSimulator(mass=0.108, scale=0.925, max_duration=0.1, plot=False, trimetric=True)
    sim.run(*state, demo=True)
