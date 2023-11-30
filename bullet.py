import pybullet as p
import numpy as np
from time import sleep


def simulate(g=1, plot=True):
    NET_MODEL = "net.obj"
    NET_EDGE_NODES = 520
    p.connect(p.GUI if plot else p.DIRECT)
    p.setGravity(0, 0, -g)
    # Net model
    netId = p.loadSoftBody(
        NET_MODEL,
        basePosition=[-1, 1, 0],
        baseOrientation=p.getQuaternionFromEuler([3.14/2, 0, 0]),
        scale=0.001, # Simulation units are meters but obj contains mm units
        mass=0.1,
        useNeoHookean=True,
        NeoHookeanMu=10,
        NeoHookeanLambda=10,
        NeoHookeanDamping=0.01,
        useSelfCollision=1,
        frictionCoeff=0.5,
        collisionMargin=0.01
    )
    # Fix rim
    perimeterNodeIndices = [*range(NET_EDGE_NODES)] # 5720 total nodes
    for nodeIndex in perimeterNodeIndices:
        p.createSoftBodyAnchor(netId, nodeIndex, -1, -1)  # Anchor to a fixed point in space
    # Ball model
    ballId = p.createMultiBody(
        baseMass=0.150, # kg
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.1), # m
        basePosition=[0, 0, 1],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    )

    ball_positions = []

    if (plot):
        p.setRealTimeSimulation(plot)
    while p.isConnected():
        if (plot):
            sleep(0.01)  # Time in seconds.
            # p.getCameraImage(320,200,renderer=p.ER_BULLET_HARDWARE_OPENGL)
        else:
            p.stepSimulation()
        # Get ball position
        ball_position, _ = p.getBasePositionAndOrientation(ballId)
        ball_positions.append(ball_position)
        # print(ball_position)
    return np.array(ball_positions)


if __name__ == "__main__":
    simulate(plot=True)
