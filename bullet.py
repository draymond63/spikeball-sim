import pybullet as p
import numpy as np
from time import sleep


def simulate(rim_contact_dist, vx, vy, g=1, max_duration=10, plot=True):
    NET_MODEL = "net.obj"
    NET_RADIUS = 0.4572 # m
    NET_EDGE_NODES = 520
    # NET_NODES = 5270
    num_steps = int(max_duration*240)
    p.connect(p.GUI if plot else p.DIRECT)
    if plot:
        # Camera settings
        cameraTargetPosition = [0, 0, 0]  # x, y, z
        cameraDistance = 3
        cameraYaw = np.pi/2
        cameraPitch = 0
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)
    p.setGravity(0, 0, -g)
    # Net model
    netId = p.loadSoftBody(
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
        collisionMargin=0.1
    )
    # print mesh data
    # vertex_coords = np.array(p.getMeshData(netId)[1])
    # x_min, x_max = np.min(vertex_coords[:, 0]), np.max(vertex_coords[:, 0])
    # y_min, y_max = np.min(vertex_coords[:, 1]), np.max(vertex_coords[:, 1])
    # z_min, z_max = np.min(vertex_coords[:, 2]), np.max(vertex_coords[:, 2])
    # print(x_min, x_max)
    # print(y_min, y_max)
    # print(z_min, z_max)

    # Fix rim
    perimeterNodeIndices = [*range(NET_EDGE_NODES)] # 5720 total nodes
    for nodeIndex in perimeterNodeIndices:
        p.createSoftBodyAnchor(netId, nodeIndex, -1, -1)  # Anchor to a fixed point in space
    # Ball model
    radius = 0.1 # m
    ballId = p.createMultiBody(
        baseMass=0.150, # kg
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=radius),
        basePosition=[-NET_RADIUS + rim_contact_dist + radius, 0, radius],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    )
    p.resetBaseVelocity(ballId, linearVelocity=[-vx, 0, -vy], angularVelocity=[0, 0, 0])

    ball_positions = []
    if plot:
        p.setRealTimeSimulation(plot)
    sim_started = False
    step_count = 0
    while p.isConnected():
        if plot:
            sleep(0.01)  # Time in seconds.
            # p.getCameraImage(320,200,renderer=p.ER_BULLET_HARDWARE_OPENGL)
        else:
            p.stepSimulation()
        step_count += 1
        # Get ball position
        ball_position, _ = p.getBasePositionAndOrientation(ballId)
        ball_positions.append(ball_position)
        if not sim_started and ball_position[2] > radius:
            sim_started = True
        elif sim_started and ball_position[2] < radius:
            break
        if step_count > num_steps:
            raise Exception("Simulation timed out")

    return np.array(ball_positions) # (step_count, 3)



if __name__ == "__main__":
    ball_coords = simulate(0.1, 3, 3, plot=False)
    print(ball_coords.shape)
