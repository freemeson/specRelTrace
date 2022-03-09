from inertialSystem import *
from emissionTimeSolver import *
from collections import namedtuple

import matplotlib.pyplot as plt

def invLength2(vec4):
    return vec4[3]*vec4[3]-vec4[0]*vec4[0]-vec4[1]*vec4[1]-vec4[2]*vec4[2]

def invLengt2Derivative(vec4, vec4_prime):
    return 2*(vec4_prime[3]*vec4[3]-vec4_prime[0]*vec4[0]-vec4_prime[1]*vec4[1]-vec4_prime[2]*vec4[2])


# class planetMoon :
#     sun_pos = np.array([0.0, 0.0, 0.0, 0.0])
#     planet_RR = 5.0
#     planet_Radius = 1.0
#     planet_omega = 0.1
#     moon_RR = 2.0
#     moon_Radius = 0.3
#     moon_omega = 0.4
#     def __init__(self, sun_pos, planet_RR, planet_Radius, planet_omega, moon_RR, moon_Radius, moon_omega):
#         self.sun_pos = np.array(sun_pos)
#         self.planet_RR = planet_RR
#         self.planet_Radius = planet_Radius
#         self.planet_omega = planet_omega
#         self.moon_RR = moon_RR
#         self.moon_Radius = moon_Radius
#         self.moon_omega = moon_omega
#
#     def planet_position(self, time):
#         return np.array([self.planet_RR*np.sin(self.planet_omega*time),
#                          self.planet_RR*np.cos(self.planet_omega*time),
#                          np.zeros_like(time),
#                          time  ], dtype=list)
#
#     def planet_velocity(self, time):
#         vel = self.planet_RR*self.planet_omega
#         return  np.array([vel*np.cos(self.planet_omega*time),
#                           -vel*np.sin(self.planet_omega*time),
#                           np.zeros_like(time) ], dtype=list)
#
#
#     def moon_position(self, time):  #will contain lorentz contraction
#         planet_pos = self.planet_position(time)
#         return np.array([self.moon_RR*np.sin(self.moon_omega*time),
#                          self.moon_RR*np.cos(self.moon_omega*time),
#                          np.zeros_like(time),
#                          np.zeros_like(time) ], dtype=list) + planet_pos
#                          #planet_pos contains time, that is not additive
#
#     def moon_velocity(self, time): #should be relativistic addition
#         vel = self.moon_RR*self.moon_omega
#         return np.array(  [vel * np.cos(self.moon_omega*time),
#                          -vel*np.sin(self.moon_omega*time),
#                          np.zeros_like(time) ], dtype=list)   + self.planet_velocity(time)
#
#     def planet_visible_pos(self, detector_4pos, t_det = None ):
#
#         t_planet = emissionTime( self.planet_omega, self.planet_RR, detector_4pos, t_det )
#         return self.planet_position( t_planet )
#
#     def moon_visible_pos(self, detector_4pos, t_det = None):
#         planet_vis_pos = self.planet_vis_pos( detector_4pos, t_det )
#         planet_vel = self.planet_velocity( planet_vis_pos[3] )
#
#     def lightConeTest( self, detector_4pos, time ):
#         pos = self.moon_position(time)
#         diff = np.array(detector_4pos) - pos
#         return invLength2(diff)
#
#     def planet_visible_iterative(self, detector_4pos, t_det = None ):
#         if t_det != None:
#             print('t_det as vector input is not implemented yet')
#
#
#         def newton_iteration( time ):
#             pos = self.planet_position(time)
#             der = np.append(self.planet_velocity(time), 1.0)
#             dist = pos - detector_4pos
#             return time - invLength2(dist)/invLengt2Derivative(dist, der)
#
#         time = detector_4pos[3] - np.linalg.norm( detector_4pos[0:3] - self.planet_position(detector_4pos[3])[0:3] )
#         for i in range(0,10):
#             time = newton_iteration(time)
#             #print(time)
#         return time
#
#     def moon_visible_iterative(self, detector_4pos, t_det = None ):
#         if t_det != None:
#             print('t_det as vector input is not implemented yet')
#
#         def newton_iteration( time ):
#             pos = self.moon_position(time)
#             der = np.append(self.moon_velocity(time), 1.0)
#             #print(pos , ' ', der)
#             dist = pos - detector_4pos
#             return time - invLength2(dist)/invLengt2Derivative(dist, der)
#
#         time = self.planet_visible_iterative(detector_4pos)
#
#         #print('planet time: ',time)#detector_4pos[3]# - np.linalg.norm( detector_4pos[0:3] - self.moon_position(detector_4pos[3])[0:3] )
#         for i in range(0,10):
#             time = newton_iteration(time)
#         #    print(time)
#         return time
#
#     def planet_apparent_inertial_system(self, detector_4pos):
#         local_time = self.planet_visible_iterative(detector_4pos)
#         planet_pos = self.planet_position(local_time)
#         planet_vel = self.planet_velocity(local_time)
#         timeShift = np.linalg.norm(detector_4pos[0:3]-planet_pos[0:3])
#         planet_pos[3] = timeShift #time needs to be faked, due to the different time in the opengl renderer
#         iS = inertialSystem(planet_pos, [1.0, 0.0, 0.0],[0.0, 1.0, 0.0],-planet_vel )
#         return iS
#
#     def moon_apparent_inertial_system(self, detector_4pos):
#         local_time = self.moon_visible_iterative(detector_4pos)
#         moon_pos = self.moon_position(local_time)
#         moon_vel = self.moon_velocity(local_time)
#         timeShift = np.linalg.norm(detector_4pos[0:3]-moon_pos[0:3])
#         moon_pos[3] = timeShift #time needs to be faked, due to the different time in the opengl renderer
#         iS = inertialSystem(moon_pos, [1.0, 0.0, 0.0],[0.0, 1.0, 0.0],-moon_vel )
#         return iS


class planetAndMoons :
    sun_pos = np.array([0.0, 0.0, 0.0, 0.0])
    sun_Radius = 1.0
    planet_RR = 5.0
    planet_Radius = 1.0
    planet_omega = 0.1
    moon_RR = [2.0]
    moon_Radius = [0.3]
    moon_omega = [0.4]
    planet_name = "Jupiter"
    E = np.eye(4)
    Einv = np.eye(4)
    def __init__(self, sun_pos, planet_RR, planet_Radius, planet_omega, moon_RR, moon_Radius, moon_omega, E = np.eye(4), planet_name = "Jupiter"):
        self.sun_pos = np.array(sun_pos)
        self.planet_RR = planet_RR
        self.planet_Radius = planet_Radius
        self.planet_omega = planet_omega
        self.moon_RR = moon_RR
        self.moon_Radius = moon_Radius
        self.moon_omega = moon_omega
        self.planet_name = planet_name
        self.E = E
        #print(E)
        self.Einv = np.linalg.inv(E)
        self.createVariableNames()

    def createVariableNames(self):
        starVars = namedtuple('starVars',['planetPos', 'planetLor', 'planetMap',
        'moonPos', 'moonLor','moonMap'])
        self.varNames = starVars(
            "planet"+self.planet_name+"position",
            "planet"+self.planet_name+"invLor",
            "planet"+self.planet_name+"texture",
            [ "planet"+self.planet_name+"Moonposition"+str(i) for i in range(0,len(self.moon_RR))],
            ["planet"+self.planet_name+"MooninvLor"+str(i) for i in range(0,len(self.moon_RR))],
            ["planet"+self.planet_name+"Moontexture"+str(i) for i in range(0,len(self.moon_RR))]
        )

    def getVariableDeclarations(self):
        uniVec4 = "uniform vec4 "
        uniMat4 = "uniform mat4 "
        uniMap = "uniform sampler2D "
        end = ";\n"
        declarations = uniVec4 + self.varNames.planetPos + end + \
            uniMat4 + self.varNames.planetLor + end + \
            uniMap + self.varNames.planetMap + end

        for i in range(len(self.varNames.moonPos)):
            declarations+=uniVec4 + self.varNames.moonPos[i] + end + \
                uniMat4 + self.varNames.moonLor[i] + end + \
                uniMap + self.varNames.moonMap[i] + end
        return declarations

    def getRayTraceCalls(self):
        moonFcn = "planetAndMoons4"
        end = ";\n"
        def callFcn(pos, lor, map, radius):
            myCalls = "dc = " + moonFcn + "(ro, rd, "+ pos +","+lor + ","+ map +","+str(radius) + " ,dlc.dLim)" + end + \
                "dlc = opU(dlc, dc.x, dc.yzw)" + end
            return myCalls

        myCalls = callFcn(self.varNames.planetPos,self.varNames.planetLor, self.varNames.planetMap , self.planet_Radius)
        for i in range(len(self.varNames.moonPos)):
            myCalls+=callFcn(self.varNames.moonPos[i],self.varNames.moonLor[i],self.varNames.moonMap[i], self.moon_Radius[i] )
        return myCalls


    def planet_position(self, time):
        return np.array([self.planet_RR*np.sin(self.planet_omega*time),
                         self.planet_RR*np.cos(self.planet_omega*time),
                         np.zeros_like(time),
                         time  ], dtype=list)

    def planet_velocity(self, time):
        vel = self.planet_RR*self.planet_omega
        return  np.array([vel*np.cos(self.planet_omega*time),
                          -vel*np.sin(self.planet_omega*time),
                          np.zeros_like(time) ], dtype=list)


    def moon_position(self, time, moon_id):  #will contain lorentz contraction
        planet_pos = self.planet_position(time)
        moon_RR = self.moon_RR[moon_id]
        moon_omega = self.moon_omega[moon_id]
        return np.array([moon_RR*np.sin(moon_omega*time),
                         moon_RR*np.cos(moon_omega*time),
                         np.zeros_like(time),
                         np.zeros_like(time) ], dtype=list) + planet_pos
                         #planet_pos contains time, that is not additive

    def moon_velocity(self, time, moon_id): #should be relativistic addition
        vel = self.moon_RR[moon_id]*self.moon_omega[moon_id]
        moon_omega = self.moon_omega[moon_id]
        return np.array(  [vel * np.cos(moon_omega*time),
                         -vel*np.sin(moon_omega*time),
                         np.zeros_like(time) ], dtype=list)   + self.planet_velocity(time)

    def planet_visible_pos(self, detector_4pos, t_det = None ):

        t_planet = emissionTime( self.planet_omega, self.planet_RR, detector_4pos, t_det )
        return self.planet_position( t_planet )


    def lightConeTest( self, detector_4pos, moon_id, time ):
        pos = self.moon_position(time, moon_id)
        diff = np.array(detector_4pos) - pos
        return invLength2(diff)

    def planet_visible_iterative(self, detector_4pos, t_det = None ):
        if t_det != None:
            print('t_det as vector input is not implemented yet')


        def newton_iteration( time ):
            pos = self.planet_position(time)
            der = np.append(self.planet_velocity(time), 1.0)
            dist = pos - detector_4pos
            return time - invLength2(dist)/invLengt2Derivative(dist, der)

        time = detector_4pos[3] - np.linalg.norm( detector_4pos[0:3] - self.planet_position(detector_4pos[3])[0:3] )
        for i in range(0,10):
            time = newton_iteration(time)
            #print(time)
        return time

    def moon_visible_iterative(self, detector_4pos, moon_id, t_det = None ):
        if t_det != None:
            print('t_det as vector input is not implemented yet')

        def newton_iteration( time ):
            pos = self.moon_position(time, moon_id)
            der = np.append(self.moon_velocity(time, moon_id), 1.0)
            #print(pos , ' ', der)
            dist = pos - detector_4pos
            return time - invLength2(dist)/invLengt2Derivative(dist, der)

        time = self.planet_visible_iterative(detector_4pos)

        #print('planet time: ',time)#detector_4pos[3]# - np.linalg.norm( detector_4pos[0:3] - self.moon_position(detector_4pos[3])[0:3] )
        for i in range(0,10):
            time = newton_iteration(time)
        #    print(time)
        return time

    def planet_apparent_inertial_system(self, detector_4pos):
        local_time = self.planet_visible_iterative(detector_4pos)
        planet_pos = self.planet_position(local_time)
        planet_vel = self.planet_velocity(local_time)
        timeShift = np.linalg.norm(detector_4pos[0:3]-planet_pos[0:3])
        planet_pos[3] = timeShift #time needs to be faked, due to the different time in the opengl renderer
        iS = inertialSystem(planet_pos, [1.0, 0.0, 0.0],[0.0, 1.0, 0.0],-planet_vel )
        return iS

    def moon_apparent_inertial_system(self, detector_4pos, moon_id = 0):
        local_time = self.moon_visible_iterative(detector_4pos, moon_id)
        moon_pos = self.moon_position(local_time, moon_id)
        moon_vel = self.moon_velocity(local_time, moon_id)
        timeShift = np.linalg.norm(detector_4pos[0:3]-moon_pos[0:3])
        moon_pos[3] = timeShift #time needs to be faked, due to the different time in the opengl renderer
        iS = inertialSystem(moon_pos, [1.0, 0.0, 0.0],[0.0, 1.0, 0.0],-moon_vel )
        return iS

def test2():
    pl = planetAndMoons([0.0, 0.0, 0.0, 0.0], 3.0, 1.0, 0.01,   [1.0,2.0,4.0], [1.0,1.0,1.0], [0.4,0.2,0.1], np.eye(4),"BELA" )
    pl.varNames.moonPos
    print(pl.getVariableDeclarations())
    print(pl.getRayTraceCalls())


def simpleTests():
    pm = planetMoon( [0.0, 0.0, 0.0, 0.0], 10.0, 1.0, 0.01,   0.2, 3.0, 0.42 )
    pm.planet_visible_pos(np.array([30., 0.0, 2.0, 0.0  ]))
    pm.planet_visible_iterative(np.array([30., 0.0, 2.0, 0.0  ]))
    pm.moon_visible_iterative(np.array([30., 0.0, 2.0, 0.0  ]))


    pm.planet_apparent_inertial_system(np.array([30., 0.0, 2.0, 0.0  ]))
    t = np.arange(-10.0, 100.0, 0.1)
    planet_pos = pm.planet_position(t)
    moon_pos = pm.moon_position(t)

    plt.plot(moon_pos[0], moon_pos[1])

    moon_vel = pm.moon_velocity(t)
    plt.plot( moon_vel[0], moon_vel[1] )


    planet_vis_pos = pm.planet_visible_pos(np.array( [30., 0.0, 2.0, 0.0  ]), t )
    plt.plot(planet_vis_pos[0], planet_vis_pos[3])

    t = np.arange(-40.0, 30.0, 0.1)
    pm.lightConeTest(np.array( [0., 0.0, 1.1, 0.0  ]), 100.0)
    i2 = [ pm.lightConeTest(np.array( [0., 1.1, 0.2, 0.0  ]), ti) for ti in t ]
    plt.plot(t, i2)
