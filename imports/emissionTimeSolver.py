import numpy as np

def emissionTime(omega, R, ro, t_det = None):

    def norm2(vec):
        return np.dot(vec, vec)
    if (type(t_det) not in [list, np.ndarray]):
        if (t_det == None):
            t_det = ro[3]

    def emissionTimeNewtonIteration( t_det, t_est  ):
        phi = np.arctan2(ro[0],ro[1])
        r = np.linalg.norm(ro[0:2])
        f_0 = (t_det - t_est  )**2 - ro[2]*ro[2] - R*R - r*r + 2*R*r*np.cos(omega*t_est-phi)
        f_0prime = 2.0*(t_est-t_det) - 2*R*r*omega*np.sin(omega*t_est - phi)
        return t_est - f_0/f_0prime
#    t_det = ro[3]
    def scalarSolve( n,re,t_d  ):
        cos_side = -np.sign(re-np.pi)

        if cos_side<0:
            n=n+0.5
        a = 1-cos_side*R*r*8*omega*omega/pisq
        b = -2*t_d + cos_side*R*r*16*omega*(phi+2*n*np.pi)/pisq
        c = cos_side*2*R*r*(1-  (4*phi*phi + 16*n*n*pisq + 16*n*np.pi*phi)/pisq ) +t_d*t_d-ro[2]*ro[2]-R*R-r*r
        discr = b*b-4*a*c
        if discr > 0.0:
            sol1 = (-b - np.sqrt(discr))/2/a
            #if np.abs(sol1)>100:
            #    sol1 = -1.0
        else:
            sol1 = np.nan# n*np.pi
        return sol1

    t2_mean = R*R+norm2(ro[0:3])
    r = np.linalg.norm(ro[0:2])
    phi = np.arctan2(ro[0],ro[1])
    pisq = np.pi*np.pi
    t_mid = t_det - np.sqrt(t2_mean)
    N,rem = np.divmod( omega*t_mid - phi+np.pi/2.0, 2.0*np.pi  )
    #print(N)
    #print(rem)
    if type(t_mid) in [list, np.ndarray]:
        sol = np.array([  scalarSolve(n,re,t_d)    for n,re,t_d in zip(N,rem, t_det)])
        sol = emissionTimeNewtonIteration(np.array(t_det), sol)
        sol = emissionTimeNewtonIteration(np.array(t_det), sol)

    else:
        sol = scalarSolve(N,rem, t_det)
        sol = emissionTimeNewtonIteration(t_det, sol)
        sol = emissionTimeNewtonIteration(t_det, sol)
    return sol
