from glumpy import app, gloo, gl, data
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



planetTilt = np.array( [[1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

plm = planetAndMoons([0.0, 0.0, 0.0, 0.0], 25.0, 2.0, 0.002,   [3.0,4.25,6.0,12], [0.3, 0.3, 0.5,0.5], [0.2,0.1,0.05, 0.025], planetTilt, "Testunus" )


fragment = """uniform sampler2D sunTexture; 
uniform sampler2D skyTexture;
uniform float cameraDirection_phi;
uniform float cameraDirection_psy;
""" + plm.getVariableDeclarations() + fragmentHeader + doppler + solids + """


vec4 sky4(in vec4 rd) {
   float len = sqrt(dot(rd.xyz,rd.xyz));

   float phi=atan(rd.z, rd.x)/M_PI/2.0+0.5;
   float theta = acos((rd.y)/len  )/M_PI;
   vec4 txcolor = texture2D(skyTexture, vec2(phi, theta));
  	vec3 red = wideSpectrum(dopplerShift(0.05, abs(rd.w)));

	vec3 green = wideSpectrum(dopplerShift(0.38, abs(rd.w)));
	vec3 blue = wideSpectrum(dopplerShift(0.71, abs(rd.w)));
//	vec3 yellow = wideSpectrum(dopplerShift(0.22, abs(rd.w)));

   vec3 color = txcolor.r*red + txcolor.g*green + txcolor.b*blue;
   
   return vec4(-1e4, color);

  
}

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

vec4 planetAndMoons4(in vec4 ro, in vec4 rd , in vec4 origin,in mat4 invLor, in sampler2D map, in float radius, in vec2 distLim   ) {
   mat4 tilted = mat4( 1.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 1.0);

   return sphereMap4(ro , rd, origin, invLor , tilted, radius, map, distLim );
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

        vec4 dc = sphereMap4(ro, rd, vec4(0.0, 0.0, 0.0, showTime), noBoost, orientation, 5.0, sunTexture, dlc.dLim);
        dlc = opU(dlc, dc.x, dc.yzw);
//        vec4 dc = box4( ro, rd, vec4(0.0, -6.0, 0.0, showTime), noBoost, orientation, vec3(2.0, 2.0, 2.0), dlc.dLim );
//	dlc = opU(dlc, dc.x, dc.yzw);

//	dc = plane4(ro, rd, vec4(0.0, -8.0, 0.0, 0.0), noBoost, tiltedOrientation, dlc.dLim );
//	dlc = opU(dlc, dc.x, dc.yzw); 
""" 

fragment += plm.getRayTraceCalls() + """
       /*dc = cylinder4(ro, rd, vec4(10.0, 0.0, 0.0, showTime), noBoost, tiltedOrientation, 3.0, 3.0, 0.2, dlc.dLim);
        dlc = opU(dlc, dc.x, dc.yzw);*/

//        dc = planet4(ro, rd, 1.0, dlc.dLim);
 //       dlc = opU(dlc, dc.x, dc.yzw);

//        dc = planetMoon4(ro, rd, vec4(0.0, 0.0, 0.0, showTime), 1.0, dlc.dLim);
//        dlc = opU(dlc, dc.x, dc.yzw);
        
        
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
       mat4 rotationCamXZ = mat4( cos(cameraDirection_phi),0.0, -sin(cameraDirection_phi), 0.0, 0.0, 1.0, 0.0, 0.0, sin(cameraDirection_phi), 0.0, cos(cameraDirection_phi), 0.0, 0.0, 0.0, 0.0, 1.0  );

	//float psy = 0.3;

	mat4 rotationYZ = mat4( 1.0, 0.0, 0.0, 0.0, 0.0, cos(psy), -sin(psy), 0.0, 0.0, sin(psy), cos(psy), 0.0, 0.0, 0.0, 0.0, 1.0 );
	mat4 rotationCamYZ = mat4( 1.0, 0.0, 0.0, 0.0, 0.0, cos(cameraDirection_psy), -sin(cameraDirection_psy), 0.0, 0.0, sin(cameraDirection_psy), cos(cameraDirection_psy), 0.0, 0.0, 0.0, 0.0, 1.0 );

	ro = camLorentz*rotationXZ * rotationYZ* ro;
	//rd = camLorentz *rotationXZ * rotationYZ*rd;
        rd = camLorentz *rotationCamXZ * rotationCamYZ*rd;

        float showTime = camDist;
        if (frozenTime!=1) {
          showTime= TIME_DEFINITION;
        }

	//dlc = worldHit(ro, rd, dlc.dLim,showTime);
dlc = testWorldHit(ro, rd, dlc.dLim,showTime);

	//gl_FragColor = vec4(dc.yzw,1.0);
	//gl_FragColor = vec4(0.5,0.5,0.5,1.0);
   if (dlc.dLim.y < 499.9) {
       gl_FragColor = vec4(dlc.color*exp(-0.016*(dlc.dLim.y-camDist)),1.0);//*frag_color;
   } else
   {  
      gl_FragColor = vec4( sky4(rd).yzw,1.0    );
   }

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

planetLorentzIS_g = inertialSystem( [0.0, 0.0, 0.0 , 0.0],  [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],   (0.0, 0.0, 0.0)  )

#planetMoonSystem = planetAndMoons([0.0, 0.0, 0.0, 0.0], 3.0, 1.0, 0.01,   [1.0,1.3], [1.0,1.0], [0.4,0.2] )

planet_apparent_direction_g = np.zeros(3)
camLorentz_g = np.eye(4, dtype=np.float32)
quad['vTexCoords0'] = texture_coords
camAngle = np.array([0.0, 0.9])
quad['phi'] = 0.0
quad['psy'] = 0.9
quad['screen_ratio'] = 1.0
quad['camLorentz'] = camLorentz_g
quad['planetLorentz'] = planetLorentzIS_g.getInvLorentzOpenGL()
quad['frozenTime'] = np.int(1)
quad[plm.varNames.planetMap] = data.get('/Users/kovesarki/src/specRelTrace/maps/jupiter_PIA02864.jpeg')
quad[plm.varNames.moonMap[0]] = data.get("/Users/kovesarki/src/specRelTrace/maps/3840px-Io_from_Galileo_and_Voyager_missions.jpeg")
quad[plm.varNames.moonMap[1]] = data.get("/Users/kovesarki/src/specRelTrace/maps/europa-IMGUR.jpeg")
quad[plm.varNames.moonMap[2]] = data.get("/Users/kovesarki/src/specRelTrace/maps/Map_of_Ganymede_by_Björn_Jónsson.jpeg")
quad[plm.varNames.moonMap[3]] = data.get("/Users/kovesarki/src/specRelTrace/maps/callisto_4k_map_by_jcp_johncarlo_dc4fjip-fullview.jpeg")
quad['sunTexture'] = data.get("/Users/kovesarki/src/specRelTrace/maps/8k_sun.jpeg")
quad['skyTexture'] = data.get("/Users/kovesarki/src/specRelTrace/maps/starmap_4k.jpeg")

target_angles = np.array([np.pi/2.0,0.0])
angvel = np.array([0.0, 0.0])
time_factor = 10.0 #the same factor is hardcoded to the glsl file
v_max = 0.99/20.0/2.0
#max acceleration
a_max = 1000.0
camLorentzSwitch = False
camKineticSwitch = False
camObserverPlanet = True

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
    iS = planetMoonSystem.moon_apparent_inertial_system(cam4D,1)
    moonInvLor = iS.getLorentzOpenGL()
    quad['moonPosition'] = iS.origin
    quad['moonInvLorentz'] = moonInvLor

def revolvingPlanetAndMoonsPosition(camPhi, camPsy, R, omega, absTime):
    def setVars(starPos, starLor, iS):
        quad[starPos] = iS.origin
        quad[starLor] = iS.getLorentzOpenGL()
        
    time = np.mod( absTime , 10.0  ) - 5.0
    cameraPos3D = sph2cart( camPhi, camPsy, 20.0 )[[1,2,0]] #GL coordinates are swapped

    #camDisplacement = -10.0*time*np.array(planetLorentzIS_g.velocity)
    cam4D = np.array( list( cameraPos3D  ) + [10.0*time]   )
    setVars(plm.varNames.planetPos, plm.varNames.planetLor, plm.planet_apparent_inertial_system(cam4D))
    for i in range(len(plm.moon_RR)):
        setVars(plm.varNames.moonPos[i], plm.varNames.moonLor[i], plm.moon_apparent_inertial_system(cam4D,i))
    
def tiltedRevolvingPlanetAndMoonsPosition(camPhi, camPsy, R, omega, absTime):
    def setVars(starPos, starLor, iS):
        quad[starPos] = plm.E.dot(iS.origin)
        quad[starLor] = plm.E.dot(iS.getLorentzOpenGL())
        
    time = absTime #np.mod( absTime , 10.0  ) - 5.0
    cameraPos3D = sph2cart( camPhi, camPsy, 20.0 )[[1,2,0]] #GL coordinates are swapped

    #camDisplacement = -10.0*time*np.array(planetLorentzIS_g.velocity)
    cam4D = plm.Einv.dot(np.array( list( cameraPos3D  ) + [10.0*time]   ))
    planet_apparent_iS =  plm.planet_apparent_inertial_system(cam4D)
    
    setVars(plm.varNames.planetPos, plm.varNames.planetLor,planet_apparent_iS)
    for i in range(len(plm.moon_RR)):
        setVars(plm.varNames.moonPos[i], plm.varNames.moonLor[i], plm.moon_apparent_inertial_system(cam4D,i))

    direction = plm.E.dot(planet_apparent_iS.origin)[0:3] - np.array(cameraPos3D)
    global planet_apparent_direction_g
    planet_apparent_direction_g = np.append(direction, -np.linalg.norm(direction))
    
#    print(plm.E.dot(planet_apparent_iS.origin)[0:4])
#    print(cameraPos3D)
#    print(normalize(direction))
    
#    loc_phi = quad['cameraDirection_phi'] =  -np.pi + np.arctan2(direction[0] , direction[2])
#    loc_psy = quad['cameraDirection_psy'] =  -np.pi/2 + np.arccos(direction[1] / np.linalg.norm(direction))
#    print(quad['cameraDirection_phi'])
#    rotM = np.array( [[np.cos(loc_phi) , 0.0, -np.sin(loc_phi)],[0.0, 1.0, 0.0], [np.sin(loc_phi),0.0, np.cos(loc_phi)]]    )
#    rotN = np.array( [np.array([1.0, 0.0, 0.0]),  np.array([0.0, np.cos(loc_psy), -np.sin(loc_psy)]), np.array([0.0, np.sin(loc_psy), np.cos(loc_psy)]) ]    )
#    print(rotM)
#    print(rotN)
#    print( rotM.dot(rotN.dot(np.array([0.0, 0.0, 1.0]))  ))
    #quad['cameraDirection_psy'] = np.arccos(direction[1] / np.linalg.norm(direction))
    

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

                quad['cameraDirection_phi'] = quad['phi']
                quad['cameraDirection_psy'] = quad['psy']

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


def observerPlanet(dt):
    angular_velocity = 0.02
    target_angles[0] += angular_velocity*dt*time_factor;
#    target_angles[1] =  0.0
    quad['phi'] = target_angles[0]
    quad['psy'] = target_angles[1]

def observerPlanetLook(dt):
    angular_velocity = 0.02

    vel = sph2cartTangent(float(quad['phi']), float(quad['psy']), float(angular_velocity), 0.0, 20.0)[[1, 2, 0]]
#    print(vel)
    iS = inertialSystem( [0.0, 0.0, 0.0 , 0.0],  [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],   vel  ) ##radius is 20.0

    global planet_apparent_direction_g
    global camLorentz_g
    camLorentz_g = iS.getLorentzOpenGL()

    #print(camLorentz_g)
    direction = camLorentz_g.dot(planet_apparent_direction_g)[0:3]
    loc_phi = quad['cameraDirection_phi'] =  -np.pi + np.arctan2(direction[0] , direction[2])
    loc_psy = quad['cameraDirection_psy'] =  -np.pi/2 + np.arccos(direction[1] / np.linalg.norm(direction))
#    print(quad['cameraDirection_phi'])
    rotM = np.array( [[np.cos(loc_phi) , 0.0, -np.sin(loc_phi)],[0.0, 1.0, 0.0], [np.sin(loc_phi),0.0, np.cos(loc_phi)]]    )
    rotN = np.array( [np.array([1.0, 0.0, 0.0]),  np.array([0.0, np.cos(loc_psy), -np.sin(loc_psy)]), np.array([0.0, np.sin(loc_psy), np.cos(loc_psy)]) ]    )

    quad['camLorentz'] = camLorentz_g
    #quad['camLorentz'] = np.eye(4)
 
def directRotation(dt):
    quad['phi'] = target_angles[0]
    quad['psy'] = target_angles[1]

    quad['cameraDirection_phi'] = quad['phi']
    quad['cameraDirection_psy'] = quad['psy']


# Tell glumpy what needs to be done at each redraw
@window.event
def on_draw(dt):
    #window.clear()
    quad['time']=app.clock.time.time()-t0
    if not camObserverPlanet:
        if camKineticSwitch:
            kineticRotation(dt)
        else:
            directRotation(dt)
    else:
        observerPlanet(dt)

       # directRotation(dt)
        
    #moonPosition(float(quad['phi']), float(quad['psy']), 3.0, 0.2, float(quad['time']))


    #freeMoonPosition(float(quad['phi']), float(quad['psy']), 2.0, 0.3, float(quad['time']))

    #revolvingMoonPosition(float(quad['phi']), float(quad['psy']), 2.0, 0.3, float(quad['time']))

    #revolvingPlanetAndMoonsPosition(float(quad['phi']), float(quad['psy']), 2.0, 0.3, float(quad['time']))
    tiltedRevolvingPlanetAndMoonsPosition(float(quad['phi']), float(quad['psy']), 2.0, 0.3, float(quad['time']))

    if camObserverPlanet:
        observerPlanetLook(dt)
        
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
