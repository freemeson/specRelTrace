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

    uniform float phi;
    uniform float psy;
    uniform float screen_ratio;
    uniform float time;
    varying vec2 tex_coord0;



//Frozen time
//#define TIME_DEFINITION camDist

//Periodic time
#define TIME_DEFINITION mod(time*10.0,camDist)+camDist/2.0
#define MAX_DIST -1e5

struct distanceAndColor{
	vec2 dLim; // distance limits
	vec3 color;
};

vec3 waveLengthToRGB(float hue)
{
    // https://www.shadertoy.com/view/ll2cDc

	return vec3(.5 + .5 * clamp( 1.3*cos(-0.3 + 6.28 * hue + vec3(0,0.66666*6.28, 0.3333*6.28)), -1.0, 1.0));
//approximate colors
//red is hue = 0.05
//green is 0.38
//blue is 0.71
}


distanceAndColor opU(distanceAndColor oldDistLim, float t, vec3 color) {

	return (-t < oldDistLim.dLim[1]) ? distanceAndColor(vec2(oldDistLim.dLim[0], -t), color) : oldDistLim;
}

float dopplerShift(float hue, float factor){
	float freq = 1.0/(hue+1.0); 
	//hue=1/freq-1
	float freqDoppl = freq*factor;
	float hueDoppl = 1.0/freqDoppl - 1.0;
	return hueDoppl;
}

float PHI = 1.61803398874989484820459;  // Φ = Golden Ratio   

float gold_noise(in vec2 xy, in float seed){
       return fract(tan(distance(xy*PHI, xy)*seed*2323.0)*xy.x);
}


vec3 wideSpectrum(float hue) {
	if (hue<0.0) {
		vec3 red = waveLengthToRGB(0.0);
	    float noise = (-hue)*gold_noise(tex_coord0, time);
		
		return red-noise*red;
    }
	if (hue>0.8) {
		vec3 violet =  waveLengthToRGB(0.8)*exp(-(hue-0.8));
		vec3 antiViolet = 1.0 - violet;
		float noise = (1.0-exp(0.16*0.8-0.16*hue))*gold_noise(tex_coord0, time*10.0); //larger seed means less correlations
		return violet+noise*antiViolet;
    }
	return waveLengthToRGB(hue);
}


vec3 wideSpectrum0(float hue) {
	if (hue<0.0) {
		vec3 red = waveLengthToRGB(0.0);
	    float noise = (1.0-exp(hue))*gold_noise(tex_coord0, time);
		
		return red-noise*red;
    }
	if (hue>0.8) {
		vec3 violet =  waveLengthToRGB(0.8);
		vec3 antiViolet = 1.0 - violet;
		float noise = (1.0-exp(0.16-0.16*hue))*gold_noise(tex_coord0, time);
		return violet+noise*antiViolet;
    }

	return waveLengthToRGB(hue);
}

vec3 wideSpectrum2(float hue) {
	if (hue<0.0) {
		vec3 red = waveLengthToRGB(0.0);
	    float noise = (-hue)*gold_noise(tex_coord0, time);
		
		return red-noise*red;
    }
	if (hue>0.8) {
		vec3 violet =  waveLengthToRGB(0.8)*exp(0.8-hue);
		vec3 antiViolet = 1.0 - violet;
		float noise = (1.0-exp(0.128-0.16*hue))*gold_noise(tex_coord0, time);
		return violet+noise*antiViolet;
    }

	return waveLengthToRGB(hue);
}


/*vec3 wideSpectrum(float hue,in vec2 xy, in float seed) {
	if (hue<0.0) {
		vec3 red = hueToRGB(0.0);
	    float noise = (-hue)*gold_noise(xy, seed);
		
		return red-noise*red;
    }
	if (hue>0.8) {
		vec3 violet =  hueToRGB(0.8)*exp(-(hue-0.8));
		vec3 antiViolet = 1.0 - violet;
		//float noise = (1.0-exp(0.16-0.16*hue))*gold_noise(xy, seed);
		float noise = (1.0-exp(0.16*0.8-0.16*hue))*gold_noise(xy, seed);
		return violet+noise*antiViolet;
//        return violet-noise*violet;
    }

	return hueToRGB(hue);
} */




vec4 sphere4(in vec4 ro, in vec4 rd, in vec4 origin, in mat4 invLor, in mat4 Einv, float radius, in vec2 distLim) {
	//origin = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 rayorig = invLor*(ro - origin);
	vec4 raydir = invLor*rd;
	
	vec4 sptq_o = Einv * rayorig;
	vec4 sptq_d = Einv * raydir;
   
	//float b = dot(rayorig.xyz , normalize(raydir.xyz));

	vec3 ray_o = sptq_o.xyz;
	float len = sqrt(dot(sptq_d.xyz,sptq_d.xyz));
	vec3 ray_d = sptq_d.xyz/len;
	float b = dot(ray_o, ray_d);	
	//radius = 9.0;
	float c = dot(ray_o, ray_o)- radius*radius;
   float discr = b*b - c;


   
   if (discr < 0.0) {return vec4(MAX_DIST, 1.0, 0.0, 0.0); }


   float t = -b+sqrt(discr);
   float ti = t/len; //assuming that the 4-vector lenght is rd*rd is zero, ligth-like

   if (-ti < distLim[0] || -ti > distLim[1]) {return vec4(MAX_DIST, 0.0, 1.0, 0.0);}

//	return vec4(0,1.0/t, 1.0, 1.0 );
	//return vec4(-t, 1.0, 1.0, 1.0);
   //the real time is t_Real = t*sptq_d.w, usually just a negative sign
   vec3 ri = ray_o + t*ray_d;
   float phi=atan(ri.z, ri.x);
   float theta = acos((ri.y)/radius);

   float shade = mod(floor(4.0*radius*phi/6.283) + floor(2.0*radius*theta/3.1415), 2.);

	vec3 red = wideSpectrum(dopplerShift(0.05, abs(sptq_d.w)));
	
	vec3 green = wideSpectrum(dopplerShift(0.38, abs(sptq_d.w)));
	vec3 blue = wideSpectrum(dopplerShift(0.71, abs(sptq_d.w)));
	vec3 yellow = wideSpectrum(dopplerShift(0.22, abs(sptq_d.w)));

   vec3 color = shade*vec3(0.0, 0.0, 0.0) + (1.0-shade)*yellow;
   return vec4(ti, color);
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
	vec3 red = wideSpectrum(dopplerShift(0.05, abs(sptq_d.w)));	
	vec3 green = wideSpectrum(dopplerShift(0.38, abs(sptq_d.w)));
	vec3 blue = wideSpectrum(dopplerShift(0.71, abs(sptq_d.w)));
	
	vec3 white = red+green+blue;
	if (abs(sptq.x) + 3e-5 > halfSizes.x) {
	
	shade = mod(floor(sptq.y) +floor(sptq.z) , 2.);
	vec3 color = shade*white + (1.0-shade)*red; 
	return vec4(tF,color);
	}

	if (abs(sptq.y) + 3e-5 > halfSizes.y) {shade = mod(floor(sptq.x) +floor(sptq.z) , 2.);
   vec3 color = shade*white + (1.0-shade)*green;
	
	return vec4(tF,color);}
	if (abs(sptq.z)+3e-5 > halfSizes.z) {shade = mod(floor(sptq.x) +floor(sptq.y) , 2.);
	//shade = 1.0;
	vec3 color = shade*white + (1.0-shade)*blue;
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
	
        mat4 orientation = mat4( 1.0, 0.0, 0.0, 0.0,
		    0.0, 1.0, 0.0, 0.0,  
		    0.0, 0.0, 1.0, 0.0,   
            0.0, 0.0, 0.0, 1.0 );

        mat4 tiltedOrientation = mat4( 1.0, 0.0, 0.0, 0.0,    0.0, 0.0, 1.0, 0.0,    0.0, 1.0, 0.0, 0.0,    0.0, 0.0, 0.0, 1.0 );

	vec4 dc = box4( ro, rd, vec4(0.0, 0.0, 0.0, showTime), invBoost05, orientation , vec3(3.0, 3.0, 3.0), dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);

	dc = box4( ro, rd, vec4(0.0, 0.0, 7.0, showTime), invBoost, orientation, vec3(3.0, 3.0, 3.0), dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);

	dc = box4( ro, rd, vec4(0.0, 0.0, -7.0, showTime), invBoost099, orientation, vec3(3.0, 3.0, 3.0), dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);
        
        
        dc = box4( ro, rd, vec4(0.0, -8.0, 0.0, showTime), noBoost, orientation, vec3(3.0, 3.0, 3.0), dlc.dLim );
	dlc = opU(dlc, dc.x, dc.yzw);

	dc = plane4(ro, rd, vec4(0.0, -11.0, 0.0, 0.0), noBoost, tiltedOrientation, dlc.dLim );	
	dlc = opU(dlc, dc.x, dc.yzw);

        dc = sphere4(ro, rd, vec4(0.0, 7.0, -7.0, showTime), invBoost, orientation, 4.0, dlc.dLim);
        dlc = opU(dlc, dc.x, dc.yzw);

        dc = sphere4(ro, rd,vec4(0.0, 7.0, 7.0, showTime), invBoost2, orientation, 3.0, dlc.dLim );
        dlc = opU(dlc, dc.x, dc.yzw);

		//dlc = distanceAndColor( vec2(0.0001, -dc.x), dc.yzw );
	return dlc;
}


void main (void){
	distanceAndColor dlc=distanceAndColor(vec2(0.0001, 500.0), vec3(0.0, 0.0, 0.0));
	
	
	float camDist = 50.0;
        float invFOV = 1.0;
	vec4 ro = vec4(0.0, 0.0, camDist, 0.0);
	vec3 rd3=normalize(vec3(tex_coord0[0]-0.5-ro.x, screen_ratio*(tex_coord0[1]-0.5-ro.y),camDist+invFOV-ro.z)); //z is funny, I know
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
