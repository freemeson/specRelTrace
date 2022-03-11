doppler = """
vec3 waveLengthToRGB(float hue)
{
    // https://www.shadertoy.com/view/ll2cDc

	return vec3(.5 + .5 * clamp( 1.3*cos(-0.3 + 6.28 * hue + vec3(0,0.66666*6.28, 0.3333*6.28)), -1.0, 1.0));
//approximate colors
//red is hue = 0.05
//green is 0.38
//blue is 0.71
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

float dopplerShift(float hue, float factor){
	float freq = 1.0/(hue+1.0);
	//hue=1/freq-1
	float freqDoppl = freq*factor;
	float hueDoppl = 1.0/freqDoppl - 1.0;
	return hueDoppl;
}


"""
