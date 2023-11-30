import pybullet as p
import numpy as np
from time import sleep

NET_RADIUS = 0.4572 # m


class SpikeBallSimulator:
    def __init__(self, max_duration=1, g=1, plot=False, trimetric=False) -> None:
        NET_MODEL = "net.obj"
        NET_EDGE_NODES = 520
        # NET_NODES = 5270
        self.TIMESTEP = 240
        self.num_steps = int(max_duration*self.TIMESTEP)
        self.plot = plot
        p.connect(p.GUI if plot else p.DIRECT)
        # Camera settings
        if plot and not trimetric:
            cameraTargetPosition = [0, 0, 0]  # x, y, z
            cameraDistance = 3
            cameraYaw = np.pi/2
            cameraPitch = 0
            p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
        p.setGravity(0, 0, -g)
        # Net model
        self.contact_margin = 0.1
        self.net = p.loadSoftBody(
            NET_MODEL,
            basePosition=[-0.5, 0.5, 0], # Center of the net is not at the origin
            baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0, 0]),
            scale=0.0005, # Model has a diameter of 1828.8. That is 6ft in mm, but we need to scale to meters
            mass=0.1,
            useNeoHookean=True,
            NeoHookeanMu=10,
            NeoHookeanLambda=10,
            NeoHookeanDamping=0.01,
            useSelfCollision=1,
            frictionCoeff=0.5,
            collisionMargin=self.contact_margin
        )
        # print mesh data
        # vertex_coords = np.array(p.getMeshData(self.net)[1])
        # x_min, x_max = np.min(vertex_coords[:, 0]), np.max(vertex_coords[:, 0])
        # y_min, y_max = np.min(vertex_coords[:, 1]), np.max(vertex_coords[:, 1])
        # z_min, z_max = np.min(vertex_coords[:, 2]), np.max(vertex_coords[:, 2])
        # print(x_min, x_max)
        # print(y_min, y_max)
        # print(z_min, z_max)

        # Fix rim
        perimeterNodeIndices = [*range(NET_EDGE_NODES)] # 5720 total nodes
        for nodeIndex in perimeterNodeIndices:
            p.createSoftBodyAnchor(self.net, nodeIndex, -1, -1)  # Anchor to a fixed point in space
        # Ball model
        self.radius = 0.1 # m
        self.ball = p.createMultiBody(
            baseMass=0.150, # kg
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius),
            basePosition=[0, 0, self.radius],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        )
        self.init_state = p.saveState()

    def run(self, rim_contact_dist, vx, vy, let_run=False):
        p.restoreState(self.init_state)
        p.resetBasePositionAndOrientation(self.ball, [NET_RADIUS - rim_contact_dist, 0, self.radius], [0, 0, 0, 1])
        p.resetBaseVelocity(self.ball, linearVelocity=[vx, 0, -vy], angularVelocity=[0, 0, 0])

        ball_coords = []
        if self.plot:
            p.setRealTimeSimulation(True)
        sim_started = False
        step_count = 0
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
            if not sim_started and z < self.radius:
                sim_started = True
            elif not let_run and sim_started and z > self.radius:
                break
            elif step_count > self.num_steps:
                if let_run:
                    break
                raise Exception("Simulation timed out")
            elif z < -3*self.radius:
                raise Exception("Ball fell through floor")

        return np.array(ball_coords) # (step_count, 3)
    
    def get_output_state(self, *input_state) -> tuple[float, float, float]:
        """Returns the output state (rim_dist, vx_out, vz_out) of the ball given the ball coordinates"""
        ball_coords = self.run(*input_state, let_run=False)
        rim_dist = NET_RADIUS - ball_coords[-1, 0]
        v_vec = p.getBaseVelocity(self.ball)[0]
        vx_out, vz_out = v_vec[0], v_vec[2]
        return rim_dist, vx_out, vz_out


if __name__ == "__main__":
    sim = SpikeBallSimulator(plot=False, trimetric=False)
    # for v in np.linspace(0, 10, 20):
    output = sim.get_output_state(0.3, 3, 3)
    print(output)
