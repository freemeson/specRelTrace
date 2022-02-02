# specRelTrace
## Ray-tracing in special relativity

Simple ray-tracing code written in python. 
The example renderer produces a ray-traced image of a checkered cube, moving with 90% the speed of light.

![moving cube](image.png?raw=true "Checkered cube moving with 0.9c")
<!-- ![plot](image.png) -->

Just run 
```
python renderMovingCube.py
```
to generate the above image as a PNG. It takes about a minute. 


## An OpenGL version

Real-time ray tracing is possible on most GPUs. This requires an openGL library and an app builder, which 
are glumpy and pyopengl in our case. Install all the requirements as

```
python -m pip install numpy pyopengl glumpy 
```

You can run
```
python renderGlumpy.py
```
for a nice moving output like this

![moving cube](imageGlumpy.png?raw=true "Checkered cube moving with 0.9c and one at rest")



## Took some inspiration from:


https://medium.com/swlh/ray-tracing-from-scratch-in-python-41670e6a96f9

https://kivy.org/doc/stable/examples/gen__demo__shadereditor__main__py.html

https://glumpy.readthedocs.io/en/latest/tutorial/easyway.html

https://www.shadertoy.com/view/tl23Rm
