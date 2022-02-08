import numpy as np

def normalize(vector):
    return vector / np.linalg.norm(vector)

class inertialSystem:
    velocity = [0,0,0]
    beta = None
    gamma = 1
    direction = None
    LorentzBoost =np.diag([1,1,1,1])
    InverseBoost=np.diag([1,1,1,1])
    origin = [0,0,0,0]
    eu=[1,0,0]
    ev=[0,1,0]
    def __init__(self,origin, eu, ev,v,beta=None):
        self.setVierBein(origin, eu, ev)
        self.setLorentzBoost(v,beta)
        
    def setVierBein(self,origin, eu, ev):
        self.origin = self._expandVector_(np.array(origin))
        self.eu = self._expandVector_(normalize(eu))
        self.ev = self._expandVector_(normalize(ev))
        self.enorm = self._expandVector_(np.cross(eu, ev))
        self.E = np.array( [[1,0,0,0], self.eu, self.ev, self.enorm ] )
        self.Einv = np.linalg.inv(self.E)
        
    def setLorentzBoost(self, v, beta=None):
        if np.linalg.norm(v) ==0:
            self.velocity=np.array([0,0,0])
            self.beta = 0
            self.direction=None
        else:    
            
            if beta==None:
                if np.linalg.norm(v)>=1:
                    print('Invalid v>c, using beta=0.5')
                    self.beta = 0.5
                    self.direction = np.array(normalize(v))
                    self.velocity =  self.beta * self.direction

                else:
                    self.beta=np.linalg.norm(v)
                    self.velocity =  np.array(v)
                    self.direction = normalize(v)
            else:
                if 0< beta <1:
                    self.beta = beta
                    self.direction = normalize(v)
                    self.velocity =  np.array(self.beta*self.direction)
                else:
                    print('invalid beta, should be 0<beta<1 or None, using beta=0.5')
                    self.beta = 0.5
                    self.direction =  np.array(normalize(v))
                    self.velocity = self.beta * self.direction
            #vnorm = np.norm(v)
            self.gamma = 1/np.sqrt(1-self.beta*self.beta)

            if np.linalg.norm(v)==0:
                self.LorentzBoost=np.diag([1,1,1,1])
                self.InverseBoost=np.diag([1,1,1,1])
            else:
                vm = np.outer(self.velocity,self.velocity)
                self.LorentzBoost = np.zeros( [4,4] )
                self.LorentzBoost[0,0] = self.gamma
                self.LorentzBoost[0,1:4] = -self.gamma * self.velocity
                self.LorentzBoost[1:4,0] = self.LorentzBoost[0,1:4]
                self.LorentzBoost[1:4,1:4] = np.diag([1,1,1]) + (self.gamma-1)*vm
                self.InverseBoost = np.linalg.inv(self.LorentzBoost)

    def getInvLorentzOpenGL(self):
        return self.matrixGLView(self.InverseBoost)

    def getLorentzOpenGL(self):
        return self.matrixGLView(self.LorentzBoost)

    @staticmethod
    def matrixGLView(mat4):
        return mat4[ [1,2,3,0],:  ][:,[1,2,3,0]]
        
    @staticmethod  
    def _expandVector_(vec):
        if len(vec) == 3:
            return np.array([0]+list(vec))
        if len(vec) == 4:
            return np.array(vec)
        print('bad vector length, len != 3')
