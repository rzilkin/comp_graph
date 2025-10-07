import numpy as np
import math
from PIL import Image, ImageOps

img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)

# def draw_line(image, x0, y0, x1, y1, count, color):
#     step = 1.0/count
#     for t in np.arange (0, 1, step):
#         x = round ((1.0 - t)*x0 + t*x1)
#         y = round ((1.0 - t)*y0 + t*y1)
#         image[y, x] = color

# def draw_line(image, x0, y0, x1, y1, count, color):
#     step = 1.0/count
#     for t in np.arange (0, 1, step):
#         x = round ((1.0 - t)*x0 + t*x1)
#         y = round ((1.0 - t)*y0 + t*y1)
#         image[y, x] = color

# def x_loop_line(image, x0, y0, x1, y1, color):
#     for x in range (x0, int(x1)):
#         t = (x-x0)/(x1 - x0)
#         y = round ((1.0 - t)*y0 + t*y1)
#         image[y, x] = color

# def x_loop_line(image, x0, y0, x1, y1, color):
#     x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#     for x in range (x0, x1):
#         t = (x-x0)/(x1 - x0)
#         y = round ((1.0 - t)*y0 + t*y1)
#         image[y, x] = color

# def x_loop_line(image, x0, y0, x1, y1, color):
#     x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

#     xchange = False
#     if (abs(x0 -x1) < abs(y0 - y1)):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True
    
#     for x in range (x0, x1):
#         t = (x-x0)/(x1 - x0)
#         y = round ((1.0 - t)*y0 + t*y1)
        
#         if (xchange):
#             image[x, y] = color
#         else:
#             image[y, x] = color

# def x_loop_line(image, x0, y0, x1, y1, color):
#     x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    

#     xchange = False
#     if (abs(x0 -x1) < abs(y0 - y1)):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True

#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0

#     for x in range (x0, x1):
#         t = (x-x0)/(x1 - x0)
#         y = round ((1.0 - t)*y0 + t*y1)
        
#         if (xchange):
#             image[x, y] = color
#         else:
#             image[y, x] = color

def x_loop_line(image, x0, y0, x1, y1, color):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    

    xchange = False
    if (abs(x0 -x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
        
    for x in range (x0, x1):
        if (xchange):
            image[x, y] = color
        else:
            image[y, x] = color
        
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update

# for k in range (13):
#     x0, y0 = 100, 100
#     x1 = 100 + 95 * math.cos(2 * np.pi * k / 13)
#     y1 = 100 + 95 * math.sin(2 * np.pi * k / 13)
#     x_loop_line(img_mat, x0, y0, x1, y1, 255)

file = open('model.obj')
f = []
v = []

for s in file:
    sp = s.split()
    if(sp[0] == 'v'):
        v.append([float(sp[1]), float(sp[2]), float(sp[3])])

    if(sp[0] == 'f'):
        f.append([int(sp[1].split('/')[0]), int(sp[2].split('/')[0]), int(sp[3].split('/')[0])])


# for vx, vy, vz in v:
    # x = int(10000 * vx + 1000)
    # y = int(10000 * vy + 1000)

#     img_mat[y, x] = (255, 255, 255)
    
for k in range(len(f)):
    x0 = v[f[k][0] - 1][0] * 9000 + 1000
    y0 = v[f[k][0] - 1][1] * 9000 + 1000
    x1 = v[f[k][1] - 1][0] * 9000 + 1000
    y1 = v[f[k][1] - 1][1] * 9000 + 1000
    x2 = v[f[k][2] - 1][0] * 9000 + 1000
    y2 = v[f[k][2] - 1][1] * 9000 + 1000

    x_loop_line(img_mat, x0, y0, x1, y1, (255, 255, 255))
    x_loop_line(img_mat, x1, y1, x2, y2, (255, 255, 255))
    x_loop_line(img_mat, x2, y2, x0, y0, (255, 255, 255))


img = ImageOps.flip(Image.fromarray(img_mat, mode='RGB'))
img.save('img.png')