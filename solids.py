solids = """
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

"""
