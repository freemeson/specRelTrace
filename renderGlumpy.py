from glumpy import app, gloo, gl
#from pyglet.window import key
from inertialSystem import *
from emissionTimeSolver import *
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
    uniform mat4 camLorentz;
    uniform vec4 moonPosition;
    uniform mat4 moonInvLorentz;
    uniform mat4 planetLorentz;

//use frozenTime to switch between periodic camera time and frozen time, this will be camDist
    uniform int frozenTime;

//Periodic time
#define TIME_DEFINITION mod(10.0*time, 100.0)+camDist-50.0
//#define TIME_DEFINITION mod(time*10.0,camDist)+camDist/2.0
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

float PHI = 1.61803398874989484820459;  

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

mat4 LorentzBoost(in vec3 v, in float beta) {
     float gamma = 1.0/sqrt(1.0-beta*beta);
     vec3 gv = gamma * v;
     float gm1 = gamma-1.0;
     return mat4( 1.0+gm1*v.x*v.x,gm1*v.x*v.y ,gm1*v.x*v.z , -gv.x ,
     	       	    gm1*v.x*v.y ,1.0+gm1*v.y*v.y ,gm1*v.y*v.z , -gv.y ,
		     gm1*v.x*v.z,gm1*v.y*v.z ,1.0+gm1*v.z*v.z , -gv.z ,
		     -gv.x,-gv.y ,-gv.z , gamma );
}



vec4 rollingSphere4(in vec4 ro, in vec4 rd, in vec4 origin, in mat4 invLor, in mat4 Einv, float radius, in float equatorSpeed, in vec2 distLim) {
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
   //the real time is t_Real = t*sptq_d.w, usually just a negative sign

   if (-ti < distLim[0] || -ti > distLim[1]) {return vec4(MAX_DIST, 0.0, 1.0, 0.0);}

//	return vec4(0,1.0/t, 1.0, 1.0 );
	//return vec4(-t, 1.0, 1.0, 1.0);

   float myTime = sptq_o.w+sptq_d.w*t;


   vec3 ri = ray_o + t*ray_d;
   float phi=atan(ri.z, ri.x);
   float theta = acos((ri.y)/radius);

   vec2 rotArrow = vec2(ri.x, ri.z);
   float rotRadius = sqrt(dot(rotArrow, rotArrow));
   vec2 rotVel = equatorSpeed*vec2(-ri.z, ri.x)/radius;


   phi += myTime*equatorSpeed/radius; 
   
   // ***** lorentz boost in 2+1 dimensions

   float beta = equatorSpeed*rotRadius/radius;
   float gamma = 1.0/sqrt(1.0-beta*beta);
   vec2 gv = gamma*rotVel;
   float gm1 = gamma - 1.0;
   /*mat3 lorentzBoostXZW = mat3( 1.0+gm1*rotVel.x*rotVel.x, gm1*rotVel.x*rotVel.y , -gv.x,
                                gm1*rotVel.x*rotVel.y,1.0+gm1*rotVel.y*rotVel.y , -gv.y,
                                -gv.x ,-gv.y , gamma);
   vec3 light2D = vec3(sptq_d.x, sptq_d.z, sptq_d.w);
   vec3 incidentLight2d = lorentzBoostXZW*light2D;
   float extraDoppler = incidentLight.w; */
   vec4 lorentzLastColumn = vec4( -gv.x, 0.0, -gv.y, gamma );
   float extraDoppler = sptq_d.w/dot(lorentzLastColumn, sptq_d);
   
   //extraDoppler = 1.0;

   float shade = mod(floor(4.0*radius*phi/6.283) + floor(2.0*radius*theta/3.1415), 2.);

	vec3 red = wideSpectrum(dopplerShift(0.05, abs(extraDoppler)));
	
	vec3 green = wideSpectrum(dopplerShift(0.38, abs(extraDoppler)));
	vec3 blue = wideSpectrum(dopplerShift(0.71, abs(extraDoppler)));
	vec3 yellow = wideSpectrum(dopplerShift(0.22, abs(extraDoppler)));

   vec3 color = shade*vec3(0.0, 0.0, 0.0) + (1.0-shade)*yellow;
   return vec4(ti, color);
}

vec4 sphere4(in vec4 ro, in vec4 rd, in vec4 origin, in mat4 invLor, in mat4 Einv, float radius, in vec2 distLim) {
    vec4 rayorig = invLor*(ro - origin);
	vec4 raydir = invLor*rd;
	
	vec4 sptq_o = Einv * rayorig;
	vec4 sptq_d = Einv * raydir;
   

	vec3 ray_o = sptq_o.xyz;
	float len = sqrt(dot(sptq_d.xyz,sptq_d.xyz));
	vec3 ray_d = sptq_d.xyz/len;
	float b = dot(ray_o, ray_d);	
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


vec4 torus4( in vec4 ro4, in vec4 rd4, in vec4 origin, in mat4 invLor, in mat4 Einv, vec2 tor, in vec2 distLim ){
    vec4 rayorig = invLor*(ro4 - origin);
    vec4 raydir = invLor*rd4;
	
    vec4 sptq_o = Einv * (rayorig);
    vec4 sptq_d = Einv * raydir;


    float sptq_d_len = sqrt(dot(sptq_d.xyz,sptq_d.xyz));
    //sptq_d = vec4(  -sptq_d.xyz/ sptq_d_len, sptq_d.w  );
    
    vec3 ro = sptq_o.xyz;
    vec3 rd = -sptq_d.xyz/sptq_d_len;    


    float po = 1.0;
    
    float Ra2 = tor.x*tor.x;
    float ra2 = tor.y*tor.y;
	
    float m = dot(ro,ro);
    float n = dot(ro,rd);

    // bounding sphere
    {
	float h = n*n - m + (tor.x+tor.y)*(tor.x+tor.y);
	if( h<0.0 ) return vec4(MAX_DIST, 0.0, 0.0, 0.0);
	//float t = -n-sqrt(h); // could use this to compute intersections from ro+t*rd
    }
    
	// find quartic equation
    float k = (m - ra2 - Ra2)/2.0;
    float k3 = n;
    float k2 = n*n + Ra2*rd.z*rd.z + k;
    float k1 = k*n + Ra2*ro.z*rd.z;
    float k0 = k*k + Ra2*ro.z*ro.z - Ra2*ra2;
	
    #if 1
    // prevent |c1| from being too close to zero
    if( abs(k3*(k3*k3 - k2) + k1) < 0.01 )
    {
        po = -1.0;
        float tmp=k1; k1=k3; k3=tmp;
        k0 = 1.0/k0;
        k1 = k1*k0;
        k2 = k2*k0;
        k3 = k3*k0;
    }
	#endif

    float c2 = 2.0*k2 - 3.0*k3*k3;
    float c1 = k3*(k3*k3 - k2) + k1;
    float c0 = k3*(k3*(-3.0*k3*k3 + 4.0*k2) - 8.0*k1) + 4.0*k0;

    
    c2 /= 3.0;
    c1 *= 2.0;
    c0 /= 3.0;
    
    float Q = c2*c2 + c0;
    float R = 3.0*c0*c2 - c2*c2*c2 - c1*c1;
    
	
    float h = R*R - Q*Q*Q;
    float z = 0.0;
    if( h < 0.0 )
    {
    	// 4 intersections
        float sQ = sqrt(Q);
        z = 2.0*sQ*cos( acos(R/(sQ*Q)) / 3.0 );
    }
    else
    {
        // 2 intersections
        float sQ = pow( sqrt(h) + abs(R), 1.0/3.0 );
        z = sign(R)*abs( sQ + Q/sQ );
    }		
    z = c2 - z;
	
    float d1 = z   - 3.0*c2;
    float d2 = z*z - 3.0*c0;
    if( abs(d1) < 1.0e-4 )
    {
        if( d2 < 0.0 ) return vec4(MAX_DIST, 0.0, 0.0, 0.0);
        d2 = sqrt(d2);
    }
    else
    {
        if( d1 < 0.0 ) return vec4(MAX_DIST, 0.0, 0.0, 0.0);
        d1 = sqrt( d1/2.0 );
        d2 = c1/d1;
    }

    //----------------------------------
	
    float result = 1e20;
    vec3 color = vec3(0.0, 0.0, 1.0);

    h = d1*d1 - z + d2;
    if( h > 0.0 )
    {
        h = sqrt(h);
        float t1 = -d1 - h - k3; t1 = (po<0.0)?2.0/t1:t1;
        float t2 = -d1 + h - k3; t2 = (po<0.0)?2.0/t2:t2;
        if( t1 > 0.0 ) {result=t1;color = vec3(1.0, 0.0, 0.0);} 
        if( t2 > 0.0 ) {result=min(result,t2);color = vec3(1.0, 0.0, 0.0);}
    }

    h = d1*d1 - z - d2;
    if( h > 0.0 )
    {
        h = sqrt(h);
        float t1 = d1 - h - k3;  t1 = (po<0.0)?2.0/t1:t1;
        float t2 = d1 + h - k3;  t2 = (po<0.0)?2.0/t2:t2;
        if( t1 > 0.0 ) {result=min(result,t1);color = vec3(1.0, 0.0, 0.0);}
        if( t2 > 0.0 ) {result=min(result,t2);color = vec3(1.0, 0.0, 0.0);}
    }


    vec3 pos = ro + result*rd;
    vec2 uv = vec2(atan(pos.x,pos.y),atan(pos.z,length(pos.xy)-tor.y));

    float shade = mod(floor(tor.x*uv.x/3.1415) + floor(2.0*tor.y*uv.y/3.1415), 2.);
    float revolveSpeed=0.3;
    float ti = sptq_o.w;
    float current_phy = ti*revolveSpeed/tor.x;

    float lagDist = (mod(current_phy-uv.x+1.570796,3.141592)-1.570796)*tor.x;//revolveSpeed;
    vec3 vel = vec3( cos(current_phy), sin(current_phy), 0.0  )*revolveSpeed;
    mat4 iLB = LorentzBoost(-vel, revolveSpeed);
    
if (shade< 0.001) {
   return sphere4(sptq_o, sptq_d, vec4( sin(current_phy)*tor.x, cos(current_phy)*tor.x,0.0, 0.0  ), iLB, Einv, 2.0, distLim  );
} else {
    return vec4(-result, color);
}

}

vec4 moon4(in vec4 ro, in vec4 rd , in float radius,  in vec2 distLim   ) {
   mat4 identity = mat4( 1.0, 0.0, 0.0, 0.0, 
                         0.0, 1.0, 0.0, 0.0, 
                         0.0, 0.0, 1.0, 0.0, 
                         0.0, 0.0, 0.0, 1.0);
   vec4 boostedRo = moonInvLorentz*ro;
   return sphere4(ro , rd, vec4(moonPosition.xyz, ro.w + moonPosition.w), moonInvLorentz , identity, radius, distLim );
}

vec4 box4rev(in vec4 ro, in vec4 rd, in vec4 origin, in mat4 invLor, in mat4 Einv, in vec3 halfSizes, in vec2 tor, in vec2 distLim  ) {
	
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
	

/*        if (abs(sptq.x) + 3e-5 > halfSizes.x) {
 	
          	shade = mod(floor(sptq.y) +floor(sptq.z) , 2.);
	        vec3 color = shade*white + (1.0-shade)*red; 
	        return vec4(tF,color);
	}

	if (abs(sptq.y) + 3e-5 > halfSizes.y) {
                shade = mod(floor(sptq.x) +floor(sptq.z) , 2.);
                vec3 color = shade*white + (1.0-shade)*green;
	        return vec4(tF,color);
        }

	if (abs(sptq.z)+3e-5 > halfSizes.z) {
               shade = mod(floor(sptq.x) +floor(sptq.y) , 2.);
	       //shade = 1.0;
	       vec3 color = shade*white + (1.0-shade)*blue;
	       //return vec4(tF,color);
        }
*/ 
        float revolveSpeed = 0.0;
        float ti = 0.0;//sptq_o.w;
        float current_phy = ti*revolveSpeed/tor.x;
        vec3 vel = vec3( -cos(current_phy), sin(current_phy), 0.0  )*revolveSpeed;
         mat4 iLB = LorentzBoost(-vel, revolveSpeed);
         mat4 noBoost = mat4(1.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0);
    
        //if (shade< 0.001) {
             return sphere4(iLB*sptq_o, iLB*sptq_d, vec4( sin(current_phy)*tor.x, cos(current_phy)*tor.x,0.0, 0.0  ), noBoost, Einv, 1.0, distLim  );
//} else {
  //           return vec4(tF, color);
  //       }




      
	
	return vec4(tF, 0.4,0.4,0.4);

	

}


vec4 torus4inst( in vec4 ro4, in vec4 rd4, in vec4 origin, in mat4 invLor, in mat4 Einv, vec2 tor, in vec2 distLim )
{
    vec4 rayorig = invLor*(ro4 - origin);
    vec4 raydir = invLor*rd4;
	
    vec4 sptq_o = Einv * (rayorig);
    vec4 sptq_d = Einv * raydir;


    float sptq_d_len = sqrt(dot(sptq_d.xyz,sptq_d.xyz));
    sptq_d = vec4(  -sptq_d.xyz/ sptq_d_len, sptq_d.w  );
    float po = 1.0;
    float Ra2 = tor.x*tor.x;
    float ra2 = tor.y*tor.y;
    float m = dot(sptq_o.xyz,sptq_o.xyz);
    float n = dot(sptq_o.xyz,sptq_d.xyz);
    float k = (m + Ra2 - ra2)/2.0;
    float k3 = n;
    float k2 = n*n - Ra2*dot(sptq_d.xy,sptq_d.xy) + k;
    float k1 = n*k - Ra2*dot(sptq_d.xy,sptq_o.xy);
    float k0 = k*k - Ra2*dot(sptq_o.xy,sptq_o.xy);
    
    if( abs(k3*(k3*k3-k2)+k1) < 0.01 )
    {
        po = -1.0;
        float tmp=k1; k1=k3; k3=tmp;
        k0 = 1.0/k0;
        k1 = k1*k0;
        k2 = k2*k0;
        k3 = k3*k0;
    }
    
    float c2 = k2*2.0 - 3.0*k3*k3;
    float c1 = k3*(k3*k3-k2)+k1;
    float c0 = k3*(k3*(c2+2.0*k2)-8.0*k1)+4.0*k0;
    c2 /= 3.0;
    c1 *= 2.0;
    c0 /= 3.0;
    float Q = c2*c2 + c0;
    float R = c2*c2*c2 - 3.0*c2*c0 + c1*c1;
    float h = R*R - Q*Q*Q;
    
    if( h>=0.0 )  
    {
        h = sqrt(h);
        float v = sign(R+h)*pow(abs(R+h),1.0/3.0); // cube root
        float u = sign(R-h)*pow(abs(R-h),1.0/3.0); // cube root
        vec2 s = vec2( (v+u)+4.0*c2, (v-u)*sqrt(3.0));
        float y = sqrt(0.5*(length(s)+s.x));
        float x = 0.5*s.y/y;
        float r = 2.0*c1/(x*x+y*y);
        float t1 =  x - r - k3; t1 = (po<0.0)?2.0/t1:t1;
        float t2 = -x - r - k3; t2 = (po<0.0)?2.0/t2:t2;
        float t = 1e20;
        if( t1>0.0 ) t=t1;
        if( t2>0.0 ) t=min(t,t2);
        return vec4(-t, 1.0, 0.0, 0.0);
    }
    
    float sQ = sqrt(Q);
    float w = sQ*cos( acos(-R/(sQ*Q)) / 3.0 );
    float d2 = -(w+c2); 
    if( d2<0.0 ) {return vec4(MAX_DIST, 0.0, 1.0, 0.0);};
    float d1 = sqrt(d2);
    float h1 = sqrt(w - 2.0*c2 + c1/d1);
    float h2 = sqrt(w - 2.0*c2 - c1/d1);
    float t1 = -d1 - h1 - k3; t1 = (po<0.0)?2.0/t1:t1;
    float t2 = -d1 + h1 - k3; t2 = (po<0.0)?2.0/t2:t2;
    float t3 =  d1 - h2 - k3; t3 = (po<0.0)?2.0/t3:t3;
    float t4 =  d1 + h2 - k3; t4 = (po<0.0)?2.0/t4:t4;
    float t = 1e20;

    vec3 color = vec3(0.0, 0.0, 1.0);
    if( t1>0.0 ) {t=t1; color = vec3(1.0, 0.0, 0.0);}
    if( t2>0.0 ) {t=min(t,t2);color = vec3(1.0, 0.0, 0.0);}
    if( t3>0.0 ) {t=min(t,t3);color = vec3(1.0, 0.0, 0.0);}
    if( t4>0.0 ) {t=min(t,t4);color = vec3(1.0, 0.0, 0.0);}
    return vec4(-t,color);
}

vec4 cylinder4(in vec4 ro, in vec4 rd, in vec4 origin, in mat4 invLor, in mat4 Einv, float radius, in float revolveRadius, in float revolveSpeed, in vec2 distLim) {
    vec4 rayorig = invLor*(ro - origin);
    vec4 raydir = invLor*rd;
	
    vec4 sptq_o = Einv * rayorig;
    vec4 sptq_d = Einv * raydir;
   
    //cylinder intersection
    vec2 Cray_o = sptq_o.xy;
    float Cray_d_len = sqrt(dot(sptq_d.xy,sptq_d.xy));
    vec2 Cray_d = sptq_d.xy/Cray_d_len;
    float Cb = dot(Cray_o, Cray_d);
    float Cc = dot(Cray_o,Cray_o) - revolveRadius*revolveRadius;//(revolveRadius+radius)*(revolveRadius+radius);
    float Cdiscr = Cb*Cb - Cc;
    if (Cdiscr < 0.0) {return vec4(MAX_DIST, 1.0, 0.0, 0.0);}

    float cT = (-Cb+sqrt(Cdiscr))/Cray_d_len;
//this is not the true distance, not even approximate, so there is no capping
    if (-cT < distLim[0] || -cT > distLim[1]) {return vec4(MAX_DIST, 0.0, 1.0, 0.0);}
    
    vec2 Cri = Cray_o + cT*Cray_d*Cray_d_len;
    float zi = sptq_o.z + sptq_d.z*cT;

    float Cphy = atan(Cri.x, Cri.y);

   float shade = mod(floor(4.0*revolveRadius*Cphy/6.283) + floor(zi), 2.);
   vec3 yellow = wideSpectrum(dopplerShift(0.22, abs(sptq_d.w)));
   vec3 color = shade*vec3(0.0, 0.0, 0.0) + (1.0-shade)*yellow;
return vec4(cT, color);


    float current_phy = cT*revolveSpeed/revolveRadius;
    float lagTime = (mod(current_phy-Cphy+1.570796,3.141592)-1.570796)*revolveRadius/revolveSpeed;
    vec2 corrected_ri = Cri * revolveRadius/(revolveRadius+radius);

//    tan phi = sin/cos=x/y -> v = (-cos, sin)
    vec3 vel = vec3( -cos(Cphy), sin(Cphy), 0.0  )*revolveSpeed;
    mat4 iLB = LorentzBoost(-vel, revolveSpeed);
    return sphere4(sptq_o, sptq_d, vec4( corrected_ri, 0.0, -cT  ), iLB, Einv, radius, distLim  );







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

	if (abs(sptq.z)+3e-5 > halfSizes.z) {
            shade = mod(floor(sptq.x) +floor(sptq.y) , 2.);
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
	
	float shade = mod(floor(uvwt.x/2.0) + floor(uvwt.y/2.0), 2.);
        vec3 red = wideSpectrum(dopplerShift(0.05, abs(uvwt_d.w)));	
	vec3 green = wideSpectrum(dopplerShift(0.38, abs(uvwt_d.w)));
	vec3 blue = wideSpectrum(dopplerShift(0.71, abs(uvwt_d.w)));
	//vec3 yellow = wideSpectrum(dopplerShift(0.22, abs(uvwt_d.w)));
	//vec3 white = red+green+blue;
        vec3 greenish = green + 0.3*red + 0.3*blue;

	return vec4(t,shade*vec3(0.0, 0.0 , 0.0) + (1.0-shade)*greenish);
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
        dc = moon4(ro, rd, 1.0, dlc.dLim);
        dlc = opU(dlc, dc.x, dc.yzw);
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
	
	//dlc = worldHit(ro, rd, dlc.dLim,showTime);
dlc = testWorldHit(ro, rd, dlc.dLim,showTime);
	
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

planetLorentzIS_g = inertialSystem( [0.0, 0.0, 0.0 , 0.0],  [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],   (0.0, 0.0, 0.2)  )

camLorentz_g = np.eye(4, dtype=np.float32)
quad['vTexCoords0'] = texture_coords
camAngle = np.array([0.0, 0.9])
quad['phi'] = 0.0
quad['psy'] = 0.9
quad['screen_ratio'] = 1.0
quad['camLorentz'] = camLorentz_g
quad['planetLorentz'] = planetLorentzIS_g.getInverseLorentzOpenGL()
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
    moonPosition(float(quad['phi']), float(quad['psy']), 3.0, 0.2, float(quad['time']))
    
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
