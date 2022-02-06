from glumpy import app, gloo, gl
import numpy as np

vertex = """
    attribute vec2 position;
    attribute vec2 vTexCoords0;
    varying vec2 tex_coord0;
    void main(){ 
      tex_coord0 = vTexCoords0;
      gl_Position = vec4(position, 0.0, 1.0); } """


fragment = """
//Frozen time
#define TIME_DEFINITION camDist

//Periodic time
//#define TIME_DEFINITION mod(time*10.0,camDist)+camDist/2.0

    uniform float phi;
    uniform float psy;
    uniform float screen_ratio;
    uniform float time;
    varying vec2 tex_coord0;
#define MAX_DIST -1e5

struct distanceAndColor{
	vec2 dLim; // distance limits
	vec3 color;
};

distanceAndColor opU(distanceAndColor oldDistLim, float t, vec3 color) {

	return (-t < oldDistLim.dLim[1]) ? distanceAndColor(vec2(oldDistLim.dLim[0], -t), color) : oldDistLim;
}


vec4 box4(in vec4 ro, in vec4 rd, in vec4 origin, in mat4 invLor, in mat4 Einv, in vec3 halfSizes, in vec2 distLim  ) {
	
       	vec4 rayorig = invLor*(ro - origin);
	vec4 raydir = invLor*rd;
	
	vec4 sptq_o = Einv * (rayorig);
	vec4 sptq_d = Einv * raydir;

	vec3 m = 1.0/sptq_d.xyz;
	vec3 n = m*sptq_o.xyz;
	vec3 k = abs(m)*halfSizes;
	vec3 t1 = -n -k;
	vec3 t2 = -n + k;
	
	float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );

	if( tN>tF || tF>0.0 || -tF<distLim[0] || -tF>distLim[1]) {
		return 	  vec4(MAX_DIST, 0,0,0);} // no intersection
	
	//return vec2(tF, 1.0 );
	vec4 sptq = sptq_o + sptq_d*tF;
	float shade = 1.0;//mod(floor(sptq.s) +floor(sptq.q) , 2.);
	//return vec4(tF,1.0, 1.0, 1.0); 
	
	if (abs(sptq.x) + 3e-5 > halfSizes.x) {
	
	shade = mod(floor(sptq.y) +floor(sptq.z) , 2.);
	vec3 color = shade*vec3(1.0, 1.0, 1.0) + (1.0-shade)*vec3(1.0, 0.0, 0.0); 
	return vec4(tF,color);
	}

	if (abs(sptq.y) + 3e-5 > halfSizes.y) {shade = mod(floor(sptq.x) +floor(sptq.z) , 2.);
   vec3 color = shade*vec3(1.0, 1.0, 1.0) + (1.0-shade)*vec3(0.0, 1.0, 0.0);
	
	return vec4(tF,color);}
	if (abs(sptq.z)+3e-5 > halfSizes.z) {shade = mod(floor(sptq.x) +floor(sptq.y) , 2.);
	//shade = 1.0;
	vec3 color = shade*vec3(1.0, 1.0, 1.0) + (1.0-shade)*vec3(0.0, 0.0, 1.0);
	return vec4(tF,color);}
	
	return vec4(tF, 0.4,0.4,0.4);

	

}

vec4 plane4(in vec4 ro, in vec4 rd, in vec4 origin, in mat4 invLor,in mat4 Einv, in vec2 distLim ){
	vec4 rayorig = invLor*(ro - origin);
	vec4 raydir = invLor*rd;
	
	vec4 uvwt_o = Einv * rayorig;
	vec4 uvwt_d = Einv * raydir;
	
	float t = -uvwt_o.z / uvwt_d.z;
	vec4 uvwt = uvwt_o + uvwt_d*t;
	
	
	if (t>0.0 || t>-distLim[0] || t<-distLim[1] ) {return vec4(MAX_DIST, 0.0, 0.0, 0.0);}
	
	float color = mod(floor(uvwt.x) + floor(uvwt.y), 2.);
	return vec4(t,color*vec3(1.0, 1.0, 1.0));
}


distanceAndColor worldHit(in vec4 ro, in vec4 rd, in vec2 distLim, in float showTime){
	distanceAndColor dlc=distanceAndColor(distLim, vec3(0.0, 0.0, 0.0));
	
	mat4 invBoost = mat4(5.26315789, 0.0, 0.0, 4.73684211,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0, 
	4.73684211, 0.0, 0.0, 4.69904779); //x-direction with 0.9c

	mat4 invBoost2 = mat4(2.29415734, 0.0, 0.0, -2.0647416, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,-2.0647416, 0.0, 0.0, 2.29415734 );
	mat4 noBoost = mat4(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0); //this is also the inverse of invBoost 
	
	vec4 dc = box4( ro, rd, vec4(0.0, 0.0, 0.0, showTime), invBoost, mat4( 1.0, 0.0, 0.0, 0.0,    0.0, 1.0, 0.0, 0.0,    0.0, 0.0, 1.0, 0.0,    0.0, 0.0, 0.0, 1.0 ), vec3(3.0, 3.0, 3.0), dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);

	dc = box4( ro, rd, vec4(0.0, -8.0, 0.0, showTime), noBoost, mat4( 1.0, 0.0, 0.0, 0.0,    0.0, 1.0, 0.0, 0.0,    0.0, 0.0, 1.0, 0.0,    0.0, 0.0, 0.0, 1.0 ), vec3(3.0, 3.0, 3.0), dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);

	dc = plane4(ro, rd, vec4(0.0, -11.0, 0.0, 0.0), noBoost,mat4( 1.0, 0.0, 0.0, 0.0,    0.0, 0.0, 1.0, 0.0,    0.0, 1.0, 0.0, 0.0,    0.0, 0.0, 0.0, 1.0 ), dlc.dLim );	
	dlc = opU(dlc, dc.x, dc.yzw);

	return dlc;
}


void main (void){
	distanceAndColor dlc=distanceAndColor(vec2(0.0001, 500.0), vec3(0.0, 0.0, 0.0));
	
	
	float camDist = 50.0;
	vec4 ro = vec4(0.0, 0.0, camDist, 0.0);
	vec3 rd3=normalize(vec3(tex_coord0[0]-0.5-ro.x, screen_ratio*(tex_coord0[1]-0.5-ro.y),camDist+1.5-ro.z)); //z is funny, I know
   vec4 rd = vec4(rd3,-1.0);
	
	//float phi = time/4.0;
   mat4 rotationXZ = mat4( cos(phi),0.0, -sin(phi), 0.0, 0.0, 1.0, 0.0, 0.0, sin(phi), 0.0, cos(phi), 0.0, 0.0, 0.0, 0.0, 1.0  );
	
	//float psy = 0.3;
	
	mat4 rotationYZ = mat4( 1.0, 0.0, 0.0, 0.0, 0.0, cos(psy), -sin(psy), 0.0, 0.0, sin(psy), cos(psy), 0.0, 0.0, 0.0, 0.0, 1.0 );

	ro = rotationXZ * rotationYZ* ro;
	rd = rotationXZ * rotationYZ*rd;
	float showTime= TIME_DEFINITION;

	
	dlc = worldHit(ro, rd, dlc.dLim,showTime);
	
	//gl_FragColor = vec4(dc.yzw,1.0);
	//gl_FragColor = vec4(0.5,0.5,0.5,1.0);
   gl_FragColor = vec4(dlc.color*exp(-0.016*(dlc.dLim.y-camDist)),1.0);//*frag_color;
}
// void main() { gl_FragColor = vec4(tex_coord0.y, 0.5*sin(time)+0.5, 0.0, 1.00); } 
"""


# Create a window with a valid GL context
window = app.Window(width = 800, height = 800)

t0 = app.clock.time.time()
# Build the program and corresponding buffers (with 4 vertices)
quad = gloo.Program(vertex, fragment, count=4)

# Upload data into GPU
quad['position'] = (-1,-1),   (-1,+1),   (+1,-1),   (+1,+1)

texture_coords = np.array([[0, 1], [0, 0], [1, 1], [1, 0]])

quad['vTexCoords0'] = texture_coords
quad['phi'] = 0.0
quad['psy'] = 0.3
quad['screen_ratio'] = 1.0

# Tell glumpy what needs to be done at each redraw
@window.event
def on_draw(dt):
    window.clear()
    quad['time']=app.clock.time.time()-t0
   # print(app.clock.time.time())
    quad.draw(gl.GL_TRIANGLE_STRIP)


@window.event
def on_resize(width, height):
    quad['screen_ratio'] = height/width;


@window.event
def on_mouse_drag(x,y,dx,dy,buttons):
   # print('drag ', x, ' ',y, ' ', dx, ' ',dy )
    #print(window.width)
    quad['phi'] += 3*dx/window.width
    quad['psy'] += 3*dy/window.height
    quad['psy'] = max(-0.1, min(0.9, quad['psy']))
    

# Run the app
app.run()
