from glumpy import app, gloo, gl
#from pyglet.window import key
from inertialSystem import *
from emissionTimeSolver import *
from orbitals import *

import numpy as np
from doppler import *
from solids import *
from fragmentHeader import *

vertex = """
    attribute vec2 position;
    attribute vec2 vTexCoords0;
    varying vec2 tex_coord0;
    void main(){
      tex_coord0 = vTexCoords0;
      gl_Position = vec4(position, 0.0, 1.0); } """



fragment = fragmentHeader + doppler + solids + """


vec4 planet4(in vec4 ro, in vec4 rd , in float radius,  in vec2 distLim   ) {
   mat4 identity = mat4( 1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0);
   vec4 boostedRo = moonInvLorentz*ro;
   return sphere4(ro , rd, vec4(moonPosition.xyz,  moonPosition.w), moonInvLorentz , identity, radius, distLim );
}

vec4 moon4(in vec4 ro, in vec4 rd , in float radius,  in vec2 distLim   ) {
   mat4 identity = mat4( 1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0);
   vec4 boostedRo = moonInvLorentz*ro;
   return sphere4(ro , rd, vec4(moonPosition.xyz, ro.w + moonPosition.w), moonInvLorentz , identity, radius, distLim );
}


vec4 planetMoon4(in vec4 ro, in vec4 rd , vec4 origin, in float radius,  in vec2 distLim   ) {
   mat4 identity = mat4( 1.0, 0.0, 0.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0);

   vec4 boostedRo = moonInvLorentz*ro;
   return sphere4(ro , rd, vec4(moonPosition.xyz, ro.w + moonPosition.w), planetLorentz*moonInvLorentz , identity, radius, distLim );
}



distanceAndColor testWorldHit(in vec4 ro, in vec4 rd, in vec2 distLim, in float showTime){
	distanceAndColor dlc=distanceAndColor(distLim, vec3(0.0, 1.0,1.0));
	mat4 invBoost = mat4(5.26315789, 0.0, 0.0, 4.73684211,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	4.73684211, 0.0, 0.0, 4.69904779); //x-direction with 0.9c

	mat4 noBoost = mat4(1.0, 0.0, 0.0,0.0,
			 0.0, 1.0, 0.0, 0.0,
			 0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0);

        mat4 orientation = mat4( 1.0, 0.0, 0.0, 0.0,
		    0.0, 1.0, 0.0, 0.0,
		    0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0 );

        mat4 tiltedOrientation = mat4( 1.0, 0.0, 0.0, 0.0,    0.0, 0.0, 1.0, 0.0,    0.0, 1.0, 0.0, 0.0,    0.0, 0.0, 0.0, 1.0 );

        vec4 dc = box4( ro, rd, vec4(0.0, -6.0, 0.0, showTime), noBoost, orientation, vec3(2.0, 2.0, 2.0), dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);

	dc = plane4(ro, rd, vec4(0.0, -8.0, 0.0, 0.0), noBoost, tiltedOrientation, dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);

       /*dc = cylinder4(ro, rd, vec4(10.0, 0.0, 0.0, showTime), noBoost, tiltedOrientation, 3.0, 3.0, 0.2, dlc.dLim);
        dlc = opU(dlc, dc.x, dc.yzw);*/

        dc = planet4(ro, rd, 1.0, dlc.dLim);
        dlc = opU(dlc, dc.x, dc.yzw);

//        dc = planetMoon4(ro, rd, vec4(0.0, 0.0, 0.0, showTime), 1.0, dlc.dLim);
//        dlc = opU(dlc, dc.x, dc.yzw);
        
        dc = sphere4(ro, rd, vec4(0.0, 0.0, 0.0, showTime), noBoost, orientation, 1.0, dlc.dLim);
        /*dc = box4rev(ro, rd, vec4(0.0, 0.0, 0.0, showTime), noBoost, orientation, vec3(5.0, 5.0, 5.0),vec2(4.0, 1.0), dlc.dLim);*/
        dlc = opU(dlc, dc.x, dc.yzw);
        return dlc;
}

distanceAndColor worldHit(in vec4 ro, in vec4 rd, in vec2 distLim, in float showTime){
	distanceAndColor dlc=distanceAndColor(distLim, vec3(0.0, 0.0, 0.0));

	mat4 invBoost = mat4(5.26315789, 0.0, 0.0, 4.73684211,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	4.73684211, 0.0, 0.0, 4.69904779); //x-direction with 0.9c

	mat4 invBoost2 = mat4(2.29415734, 0.0, 0.0, -2.0647416, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,-2.0647416, 0.0, 0.0, 2.29415734 ); //this is also the inverse of invBoost

	mat4 invBoost05 = mat4(1.3333333, 0.0, 0.0, 0.666666,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		0.666666, 0.0, 0.0, 1.19935874); //half the speed of ligth
	mat4 noBoost = mat4(1.0, 0.0, 0.0,0.0,
			 0.0, 1.0, 0.0, 0.0,
			 0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0);

        mat4 invBoost099 = mat4(50.25125628,  0.        ,  0.        , 49.74874372,
          0.        ,  1.        ,  0.        ,  0.        ,
          0.        ,  0.        ,  1.        ,  0.        ,
          49.74874372,  0.        ,  0.        , 49.39232364);

        mat4 invBoost01 = mat4(1.010101 , 0.       , 0.       , 0.1010101,
       0.       , 1.       , 0.       , 0.       ,
       0.       , 0.       , 1.       , 0.       ,
       0.1010101, 0.       , 0.       , 1.0050884   );

        mat4 orientation = mat4( 1.0, 0.0, 0.0, 0.0,
		    0.0, 1.0, 0.0, 0.0,
		    0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0 );

        mat4 tiltedOrientation = mat4( 1.0, 0.0, 0.0, 0.0,    0.0, 0.0, 1.0, 0.0,    0.0, 1.0, 0.0, 0.0,    0.0, 0.0, 0.0, 1.0 );
        mat4 tiltedOrientationRoll = mat4( 1.0, 0.0, 0.0, 0.0,
                                           0.0, 0.0, -1.0, 0.0,
                                           0.0, -1.0, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 1.0 );



	vec4 dc = box4( ro, rd, vec4(0.0, 0.0, 0.0, showTime), invBoost, orientation , vec3(2.0, 2.0, 2.0), dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);

	dc = box4( ro, rd, vec4(0.0, 0.0, 5.0, showTime), invBoost05, orientation, vec3(2.0, 2.0, 2.0), dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);

	dc = box4( ro, rd, vec4(0.0, 0.0, -5.0, showTime), invBoost099, orientation, vec3(2.0, 2.0, 2.0), dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);


        dc = box4( ro, rd, vec4(0.0, -6.0, 0.0, showTime), noBoost, orientation, vec3(2.0, 2.0, 2.0), dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);

	dc = plane4(ro, rd, vec4(0.0, -8.0, 0.0, 0.0), noBoost, tiltedOrientation, dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);

        dc = sphere4(ro, rd, vec4(0.0, 0.0, -9.0, showTime), invBoost, orientation, 2.0, dlc.dLim);
        dlc = opU(dlc, dc.x, dc.yzw);

        dc = sphere4(ro, rd,vec4(0.0, 0.0, 9.0, showTime), invBoost2, orientation, 2.0, dlc.dLim );
        dlc = opU(dlc, dc.x, dc.yzw);

        dc = sphere4(ro, rd, vec4(0.0, -6.0, -5.0, showTime), noBoost, orientation, 2.0, dlc.dLim);
        dlc = opU(dlc, dc.x, dc.yzw);

        dc = rollingSphere4(ro, rd, vec4(0.0, -6.0, -10.0, showTime), noBoost, orientation, 2.0, 0.1, dlc.dLim);
        dlc = opU(dlc, dc.x, dc.yzw);
        dc = rollingSphere4(ro, rd, vec4(0.0, -6.0, 10.0, showTime), invBoost05, tiltedOrientationRoll, 2.0, 0.5, dlc.dLim);
        dlc = opU(dlc, dc.x, dc.yzw);


		//dlc = distanceAndColor( vec2(0.0001, -dc.x), dc.yzw );
	return dlc;
}


void main (void){
	distanceAndColor dlc=distanceAndColor(vec2(0.0001, 500.0), vec3(0.0, 0.0, 0.0));


	float camDist = 20.0;
        float invFOV = 0.5;
	vec4 ro = vec4(0.0, 0.0, camDist, 0.0);
	vec3 rd3=normalize(vec3(tex_coord0[0]-0.5-ro.x, screen_ratio*(tex_coord0[1]-0.5-ro.y),camDist+invFOV-ro.z)); //z is funny, I know
   vec4 rd = vec4(rd3,-1.0);

	//float phi = time/4.0;
   mat4 rotationXZ = mat4( cos(phi),0.0, -sin(phi), 0.0, 0.0, 1.0, 0.0, 0.0, sin(phi), 0.0, cos(phi), 0.0, 0.0, 0.0, 0.0, 1.0  );

	//float psy = 0.3;

	mat4 rotationYZ = mat4( 1.0, 0.0, 0.0, 0.0, 0.0, cos(psy), -sin(psy), 0.0, 0.0, sin(psy), cos(psy), 0.0, 0.0, 0.0, 0.0, 1.0 );

	ro = camLorentz*rotationXZ * rotationYZ* ro;
	rd = camLorentz *rotationXZ * rotationYZ*rd;

        float showTime = camDist;
        if (frozenTime!=1) {
          showTime= TIME_DEFINITION;
        }

	dlc = worldHit(ro, rd, dlc.dLim,showTime);
        //dlc = testWorldHit(ro, rd, dlc.dLim,showTime);

	//gl_FragColor = vec4(dc.yzw,1.0);
	//gl_FragColor = vec4(0.5,0.5,0.5,1.0);
   gl_FragColor = vec4(dlc.color*exp(-0.016*(dlc.dLim.y-camDist)),1.0);//*frag_color;
}



"""


# Create a window with a valid GL context
window = app.Window(width = 800, height = 800)

t0 = app.clock.time.time()
# Build the program and corresponding buffers (with 4 vertices)
quad = gloo.Program(vertex, fragment, count=4)

# Upload data into GPU
quad['position'] = (-1,-1),   (-1,+1),   (+1,-1),   (+1,+1)

texture_coords = np.array([[0, 1], [0, 0], [1, 1], [1, 0]])

#planetLorentzIS_g = inertialSystem( [0.0, 0.0, 0.0 , 0.0],  [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],   (0.0, 0.0, 0.0)  )

#planetMoonSystem = planetMoon([0.0, 0.0, 0.0, 0.0], 3.0, 1.0, 0.01,   1.0, 3.0, 0.4 )

camLorentz_g = np.eye(4, dtype=np.float32)
quad['vTexCoords0'] = texture_coords
camAngle = np.array([0.0, 0.9])
quad['phi'] = 0.0
quad['psy'] = 0.9
quad['screen_ratio'] = 1.0
quad['camLorentz'] = camLorentz_g
#quad['planetLorentz'] = planetLorentzIS_g.getInvLorentzOpenGL()
quad['frozenTime'] = np.int(1)

target_angles = np.array([np.pi/2.0,0.0])
angvel = np.array([0.0, 0.0])
time_factor = 10.0 #the same factor is hardcoded to the glsl file
v_max = 0.99/20.0/2.0
#max acceleration
a_max = 1000.0
camLorentzSwitch = True
camKineticSwitch = True

def camMessage(camLorentzSwitch, camKineticSwitch):
    if quad['frozenTime']:
        print('Camera time is frozen at camera distance to origin')
    else:
        print('Camera time is periodic')

    if camLorentzSwitch and camKineticSwitch:
        print('Camera movement is kinetic and Lorentz boost is physical')

    if camLorentzSwitch and not camKineticSwitch:
        print('Camera Lorentz boost is frozen, camera movement is kinetic but unphysical')

    if not camLorentzSwitch: #and not camKineticSwitch:
        print('Camrea Lorentz boost is off, camera movement is unphysical')

#    if not camLorentzSwitch #and camKineticSwitch:
#        print('Camera Lorenzt boost is off, camera movement is kinetic, but unphysical')

def printHelp():
    print('-------------------------------------------------------------------------')
    camMessage(camLorentzSwitch, camKineticSwitch)
    print('press \'c\' to switch camera movement between instantaneous and kinetic ')
    print('press \'l\' to switch camera movement\'s Lorenzt Boost on or off')
    print('press \'t\' to switch camera time between frozen and periodic')
    print('click and drag with the mouse pointer to rotate screen')

    print('\npress \'h\' for this help message')
    print('-------------------------------------------------------------------------')


printHelp()

def normalize(vec):
    return np.array(vec)/np.linalg.norm(vec)


def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return np.array([x, y, z])

def sph2cartTangent(az, el,  dAz, dEl, r ):
    rcos_theta_prime = -r * np.sin(el)
    rcos_theta = r * np.cos(el)
    saz = np.sin(az)
    caz = np.cos(az)

    grad_el = np.array([ rcos_theta_prime*caz, -rcos_theta_prime*saz, rcos_theta  ])

    grad_az = np.array([rcos_theta*saz, -rcos_theta*caz, 0.0 ])
    return (grad_el*dEl - grad_az*dAz)

def moonPosition(camPhi, camPsy, R, omega, time):
    cameraPos3D = sph2cart( camPhi, camPsy, 20.0 )[[1,2,0]] #GL coordinates are swapped
    cam4D = np.array( list( cameraPos3D ) + [10.0*time]   )
    #print(cam4D)
    global camLorentz_g
    #_,t_moon = np.divmod(emissionTime(omega, R, camLorentz_g.dot(cam4D)), 2*np.pi/omega  )
    #t_moon = emissionTime(omega, R, camLorentz_g.dot(cam4D))
    t_moon = emissionTime(omega, R, cam4D)
#    print(t_moon)
    #vel = np.array([0.0, 0.8, 0.0])
    vel = np.array( [omega*R*np.cos(omega*t_moon),-omega*R*np.sin(omega*t_moon), 0.0]  )*0.9
    #vel4 = np.array([vel[0],vel[1],vel[2],-1.0])
    vel4 = np.array([0.0, 0.0 ,0.0,-1.0])
    pos = np.array( [R*np.sin(omega*t_moon), R*np.cos(omega*t_moon) , 0.0, 0.0]  )

    #print(quad['moonPosition'])

    iS = inertialSystem( [0.0, 0.0, 0.0 , 0.0],  [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],   vel  )
    moonInvLor = iS.getInvLorentzOpenGL()
    moonLor = iS.getLorentzOpenGL()
    cam4D[3] = 20.0 #
#    camAberration = moonLor.dot( camLorentz_g.dot(cam4D) )
    #somehow this transform is not needed
    camAberration = cam4D#( camLorentz_g.dot(cam4D) )

    timeShift = np.linalg.norm(camAberration[0:3]-pos[0:3])
#    print(timeShift)
    quad['moonPosition'] = pos - vel4*timeShift
    quad['moonInvLorentz'] = moonInvLor

def revolvingPlanetPosition(camPhi, camPsy, R, omega, absTime):
    time = np.mod( absTime , 10.0  ) - 5.0
    cameraPos3D = sph2cart( camPhi, camPsy, 20.0 )[[1,2,0]] #GL coordinates are swapped

    #camDisplacement = -10.0*time*np.array(planetLorentzIS_g.velocity)
    cam4D = np.array( list( cameraPos3D  ) + [10.0*time]   )
    iS = planetMoonSystem.planet_apparent_inertial_system(cam4D)
    moonInvLor = iS.getLorentzOpenGL()
    quad['moonPosition'] = iS.origin
    quad['moonInvLorentz'] = moonInvLor

def revolvingMoonPosition(camPhi, camPsy, R, omega, absTime):
    time = np.mod( absTime , 10.0  ) - 5.0
    cameraPos3D = sph2cart( camPhi, camPsy, 20.0 )[[1,2,0]] #GL coordinates are swapped

    #camDisplacement = -10.0*time*np.array(planetLorentzIS_g.velocity)
    cam4D = np.array( list( cameraPos3D  ) + [10.0*time]   )
    iS = planetMoonSystem.moon_apparent_inertial_system(cam4D)
    moonInvLor = iS.getLorentzOpenGL()
    quad['moonPosition'] = iS.origin
    quad['moonInvLorentz'] = moonInvLor

def freeMoonPosition(camPhi, camPsy, R, omega, absTime):
    time = np.mod( absTime , 10.0  ) - 5.0
    cameraPos3D = sph2cart( camPhi, camPsy, 20.0 )[[1,2,0]] #GL coordinates are swapped
    global camLorentz_g
    global planetLorentzIS_g
    camDisplacement = -10.0*time*np.array(planetLorentzIS_g.velocity)
    cam4D = np.array( list( cameraPos3D + camDisplacement ) + [10.0*time]   )

    t_moon = emissionTime(omega, R, planetLorentzIS_g.getInvLorentzOpenGL().dot(cam4D) )
    vel = np.array( [omega*R*np.cos(omega*t_moon),-omega*R*np.sin(omega*t_moon), 0.0]  )*0.9
    vel4 = np.array([0.0, 0.0 ,0.0,-1.0])
    pos = np.array( [R*np.sin(omega*t_moon), R*np.cos(omega*t_moon) , 0.0, 0.0]  )
    iS = inertialSystem( [0.0, 0.0, 0.0 , 0.0],  [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],   vel  )
    moonInvLor = iS.getInvLorentzOpenGL()
    moonLor = iS.getLorentzOpenGL()
    cam4D[3] = 20.0 #
    camAberration = (cam4D)#( camLorentz_g.dot(cam4D) )

    timeShift = np.linalg.norm(camAberration[0:3]-pos[0:3])
    quad['moonPosition'] = pos- np.append(camDisplacement,0.0) - vel4*timeShift
    quad['moonInvLorentz'] = moonInvLor

     
def kineticRotation(dt):
    global angvel
    global camLorentzSwitch
    global camLorentz_g
    # print(app.clock.time.time())
    if (target_angles[0] != quad['phi']) or (target_angles[1]!=quad['psy']) or (angvel[0] != 0.0) or (angvel[1] != 0.0) :
        arc_diff = target_angles - np.array( [ float(quad['phi']), float(quad['psy']) ]  )

        angvel *= 0.95 #drag
        diff_norm = np.linalg.norm(arc_diff)
        if diff_norm != 0:
            new_norm = min(a_max, diff_norm)
            acc = arc_diff*(new_norm/diff_norm)
            new_angular_velocity = angvel+acc*dt
            new_speed = np.linalg.norm(new_angular_velocity)
            if new_speed != 0:
                maximized_speed = min(v_max, new_speed)
                new_angular_velocity *= maximized_speed/new_speed
                angvel = new_angular_velocity
                #new position

                quad['phi'] += new_angular_velocity[0]*dt*time_factor
                quad['psy'] += new_angular_velocity[1]*dt*time_factor
                #print(angvel)
#                pos = sph2cart(psi, phy, 20.0)[ 0, 2, 1  ] #x,z,y is needed
                vel = sph2cartTangent(float(quad['phi']), float(quad['psy']), float(new_angular_velocity[0]), float(new_angular_velocity[1]), 20.0)[[1, 2, 0]]
                velnorm = np.linalg.norm(vel)
                #print(vel)
                #print(new_angular_velocity)
                iS = inertialSystem( [0.0, 0.0, 0.0 , 0.0],  [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],   vel  ) ##radius is 20.0
                if camLorentzSwitch:
                    camLorentz_g = iS.getLorentzOpenGL()
                else:
                    camLorentz_g = np.eye(4, dtype=np.float32)

                quad['camLorentz'] = camLorentz_g
            else:
                camLorentz_g = np.eye(4, dtype=np.float32)
                quad['camLorentz'] = camLorentz_g


def directRotation(dt):
    quad['phi'] = target_angles[0]
    quad['psy'] = target_angles[1]

# Tell glumpy what needs to be done at each redraw
@window.event
def on_draw(dt):
    #window.clear()
    quad['time']=app.clock.time.time()-t0
    if camKineticSwitch:
        kineticRotation(dt)
    else:
        directRotation(dt)
    #moonPosition(float(quad['phi']), float(quad['psy']), 3.0, 0.2, float(quad['time']))


    #freeMoonPosition(float(quad['phi']), float(quad['psy']), 2.0, 0.3, float(quad['time']))
    #revolvingMoonPosition(float(quad['phi']), float(quad['psy']), 2.0, 0.3, float(quad['time']))
    quad.draw(gl.GL_TRIANGLE_STRIP)


@window.event
def on_resize(width, height):
    quad['screen_ratio'] = height/width;

@window.event
def on_key_press(symbol, modifiers):
    print('Key pressed (symbol=%s, modifiers=%s)'% (symbol,modifiers))
    if symbol==97:
        target_angles[0] += 0.01
    if symbol==100:
        target_angles[0] -= 0.01

    if symbol==108:
        global camLorentzSwitch
        camLorentzSwitch = not camLorentzSwitch

    if symbol==99:
        global camKineticSwitch
        camKineticSwitch = not camKineticSwitch
    if symbol==116:
        quad['frozenTime'] = not quad['frozenTime']
        #'l'=108, 'c'= 99, 't'= 116
    if symbol==104:
        printHelp()

    camMessage(camLorentzSwitch,camKineticSwitch )

@window.event
def on_mouse_drag(x,y,dx,dy,buttons):
   # print('drag ', x, ' ',y, ' ', dx, ' ',dy )
    #print(window.width)
    target_angles[0] += 3*dx/window.width
    target_angles[1] += 3*dy/window.height
    target_angles[1] = max(-0.1, min(1.5, target_angles[1]))
    #quad['phi'] += 3*dx/window.width
    #quad['psy'] += 3*dy/window.height
    #quad['psy'] = max(-0.1, min(0.9, quad['psy']))


# Run the app
app.run()
