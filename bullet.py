import pybullet as p
import numpy as np
from time import sleep

NET_RADIUS = 0.4572 # m


class SpikeBallSimulator:
    def __init__(self, max_duration=1, g=1, plot=False, trimetric=False) -> None:
        NET_MODEL = "net.obj"
        NET_EDGE_NODES = 520
        # NET_NODES = 5270
        TIMESTEP = 240
        self.num_steps = int(max_duration*TIMESTEP)
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
        self.base_net_position = [-0.5, 0.5, 0]
        self.base_net_orientation = p.getQuaternionFromEuler([np.pi/2, 0, 0])
        self.net = p.loadSoftBody(
            NET_MODEL,
            basePosition=self.base_net_position, # Center of the net is not at the origin
            baseOrientation=self.base_net_orientation,
            scale=0.0005, # Model has a diameter of 1828.8. That is 6ft in mm, but we need to scale to meters
            mass=0.1,
            useNeoHookean=True,
            NeoHookeanMu=10,
            NeoHookeanLambda=10,
            NeoHookeanDamping=0.01,
            useSelfCollision=1,
            frictionCoeff=0.5,
            collisionMargin=0.1
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

    def run(self, rim_contact_dist, vx, vy):
        p.resetBasePositionAndOrientation(self.net, [0, 0, 0], self.base_net_orientation)
        p.resetBasePositionAndOrientation(self.ball, [-NET_RADIUS + self.radius + rim_contact_dist, 0, self.radius], [0, 0, 0, 1])
        p.resetBaseVelocity(self.ball, linearVelocity=[-vx, 0, -vy], angularVelocity=[0, 0, 0])

        ball_positions = []
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
            ball_position, _ = p.getBasePositionAndOrientation(self.ball)
            ball_positions.append(ball_position)
            if not sim_started and ball_position[2] < self.radius:
                sim_started = True
            elif sim_started and ball_position[2] > self.radius:
                break
            if step_count > self.num_steps:
                raise Exception("Simulation timed out")

        return np.array(ball_positions) # (step_count, 3)



if __name__ == "__main__":
    sim = SpikeBallSimulator(plot=True, trimetric=False)
    for v in np.linspace(0, 10, 20):
        ball_coords = sim.run(0.1, v, v)
        print(ball_coords.shape)
