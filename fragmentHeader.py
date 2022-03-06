fragmentHeader = """
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



distanceAndColor opU(distanceAndColor oldDistLim, float t, vec3 color) {

	return (-t < oldDistLim.dLim[1]) ? distanceAndColor(vec2(oldDistLim.dLim[0], -t), color) : oldDistLim;
}
"""
