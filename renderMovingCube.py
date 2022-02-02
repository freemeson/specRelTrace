from cube import *
import matplotlib.pyplot as plt


width = 600;height = 400
#width = 300;height = 200

camera = np.array([0, 0, 1])
ratio = float(width) / height
fa = 1
screen = (-fa, fa / ratio, fa, -fa / ratio) # left, top, right, bottom

image = np.zeros((height, width, 3))

cube =  Cube(6, np.array([0,-30,0,-6]),np.array([0.5,0.0,  0.00]),np.array([0.0, 0.5, 0.0]), [1,0.0,0], 0.90)
#cube =  Cube(6, np.array([0,-50,0,-6]),np.array([0.5,0.0,  0.00]),np.array([0.0, 0.5, 0.0]), [1,0.0,0], 0.95)
#cube =  Cube(6, np.array([0,-300,0,-6]),np.array([0.5,0.0,  0.00]),np.array([0.0, 0.5, 0.0]), [1,0.0,0], 0.99)

for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        pixel = np.array([0,x, y, 0])
        origin = [0,camera[0],camera[1],camera[2]]
        direction = normalize(pixel - origin)
        direction[0] = -1
        color = cube.intersection(origin, direction)
        #color = np.ones(3)
        if color==None:
            image[i, j] = (0,0,0)
            continue
        #print(coord)

        image[i,j] = np.clip( color, 0, 1)
        
        
        # image[i, j] = ...

    print("progress: %d/%d" % (i + 1, height))

plt.imsave('image.png', image)
