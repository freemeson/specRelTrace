from inertialSystem import *
import numpy as np

class Cube:
    sideColors = [(0.3,0,0), (0,0.3,0), (0,0,0.3)  ]
    def __init__(self, size, center, eu, ev,vel, beta = None):
        self.orientation = inertialSystem(center, eu, ev, vel, beta)
        self.halfSize = size/2
        self.sides = [ (1,-self.halfSize),(1,self.halfSize),(2,-self.halfSize),(2,self.halfSize),(3,-self.halfSize),(3,self.halfSize), ]

    def intersection(self, ray_origin, ray_direction):
        ray_orig = self.orientation._expandVector_(ray_origin)

        ray_dir = self.orientation._expandVector_(ray_direction)
       # print(ray_direction)
        rayo = self.orientation.InverseBoost.dot(ray_orig) - self.orientation.origin
        tuvw_o = self.orientation.Einv.dot(rayo)
        tuvw_d = self.orientation.Einv.dot(self.orientation.InverseBoost.dot(ray_dir))
        intersections = [np.array(np.zeros(4))] * 6
        closest = -np.inf
        closest_index = -1
        
        for i,side in enumerate(self.sides):
            t = -(tuvw_o[side[0]]+side[1])/tuvw_d[side[0]]
            intersections[i] = tuvw_o + t*tuvw_d
            if sum(abs(intersections[i][1:4])<self.halfSize+1e-5) == 3: 
                if intersections[i][0] < 0 and closest < intersections[i][0]: # -inf is past, 0 is now, +inf is future (or backward ray)
                    closest = intersections[i][0]
                    closest_index = i
            

        if closest_index == -1:
            return None
        tuvw = intersections[closest_index]
        surface_index = self.sides[closest_index][0]
        surface_ui = (surface_index) % 3 +1
        surface_vi = (surface_index+1) % 3 +1
        #print(surface_index, ": " , surface_ui, "  " , surface_vi)
        
        if int(np.floor(tuvw[surface_ui])+np.floor(tuvw[surface_vi ])) % 2 == 1:
            return (1,1,1)
        else:
            return self.sideColors[surface_index % 3]




