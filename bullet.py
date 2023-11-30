# import pybullet as p
# import time
# import pybullet_data

# # Start the physics client
# physicsClient = p.connect(p.DIRECT) # p.GUI

# # Add pybullet_data to the search path
# # p.setAdditionalSearchPath(pybullet_data.getDataPath())
# # Load in the standard plane
# # planeId = p.loadURDF("plane.urdf")

# # Create a soft body net
# # https://www.reddit.com/r/SolidWorks/comments/12fsrmn/no_available_entities_to_process_through_wrl/
# netID = p.loadSoftBody("net.obj", basePosition=[0, 0, 1], scale=1, mass=0.1, useNeoHookean=1, NeoHookeanMu=180, NeoHookeanLambda=600, NeoHookeanDamping=0.01, useSelfCollision=1, frictionCoeff=0.5, collisionMargin=0.001)

# meshData = p.getMeshData(netID)
# print("meshData=",meshData)

# # Create a ball
# # ballRadius = 0.1 # m
# # ballMass = 0.150 # kg
# # ballCollisionShapeId = p.createCollisionShape(p.GEOM_SPHERE, radius=ballRadius)
# # ballVisualShapeId = -1  # Use default visual shape
# # ballStartPosition = [0, 0, 2]
# # ballStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
# # ballId = p.createMultiBody(ballMass, ballCollisionShapeId, ballVisualShapeId, ballStartPosition, ballStartOrientation)

# # # Set the gravity
# # p.setGravity(0, 0, -10)

# # # Simulation loop
# # for i in range(10000):
# #     p.stepSimulation()
# #     time.sleep(1./240.)

# # Disconnect the physics client
# p.disconnect()


import pybullet as p
from time import sleep
import pybullet_data

physicsClient = p.connect(p.GUI)
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf", [0, 0, 0])
boxId = p.loadURDF("cube.urdf", [0, 0, 2])
bunnyId = p.loadSoftBody("net.obj", basePosition=[2, -1, 0], scale=0.001, mass=0.1, useNeoHookean=1, NeoHookeanMu=180, NeoHookeanLambda=600, NeoHookeanDamping=0.01, useSelfCollision=1, frictionCoeff=0.5, collisionMargin=0.001)#.obj")#.vtk")

#meshData = p.getMeshData(bunnyId)
#print("meshData=",meshData)
#p.loadURDF("cube_small.urdf", [1, 0, 1])
useRealTimeSimulation = 1

if (useRealTimeSimulation):
  p.setRealTimeSimulation(1)

print("planeId", p.getDynamicsInfo(planeId, -1))
print("bunnyId", p.getDynamicsInfo(bunnyId, 0))
print("boxId", p.getDynamicsInfo(boxId, -1))
p.changeDynamics(boxId,-1,mass=10)
while p.isConnected():
  p.setGravity(0, 0, -10)
  if (useRealTimeSimulation):

    sleep(0.01)  # Time in seconds.
    #p.getCameraImage(320,200,renderer=p.ER_BULLET_HARDWARE_OPENGL )
  else:
    p.stepSimulation()