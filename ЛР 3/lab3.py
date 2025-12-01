import numpy as np
from PIL import Image, ImageOps


def calc_barCord(x, y, x0, y0, x1, y1, x2, y2):
    den = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    if den == 0:
        return (-1, -1, -1)
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / den
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / den
    lambda2 = 1.0 - lambda0 - lambda1
    return (lambda0, lambda1, lambda2)


def calc_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    return np.cross((x1 - x2, y1 - y2, z1 - z2), (x1 - x0, y1 - y0, z1 - z0))


def draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, z_buf, matrix, color):
    H, W, _ = matrix.shape
    xmin = int(min(x0, x1, x2))
    xmax = int(max(x0, x1, x2))
    ymin = int(min(y0, y1, y2))
    ymax = int(max(y0, y1, y2))

    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax >= W:
        xmax = W - 1
    if ymax >= H:
        ymax = H - 1
    if xmin > xmax or ymin > ymax:
        return

    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            l0, l1, l2 = calc_barCord(x, y, x0, y0, x1, y1, x2, y2)
            if l0 >= 0 and l1 >= 0 and l2 >= 0:
                z_hat = l0 * z0 + l1 * z1 + l2 * z2
                if z_hat < z_buf[y, x]:
                    z_buf[y, x] = z_hat
                    matrix[y, x] = color


def rot_matrix(alpha, beta, gamma):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    Rx = np.array([[1, 0, 0], [0, ca, sa], [0, -sa, ca]], dtype=np.float64)

    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float64)

    Rz = np.array([[cg, sg, 0], [-sg, cg, 0], [0, 0, 1]], dtype=np.float64)
    return Rx @ Ry @ Rz


H, W = 2000, 2000
img_mat = np.zeros((H, W, 3), dtype=np.uint8)
zbuf = np.full((H, W), np.inf, dtype=np.float64)

file = open("model.obj", "r")
lines = file.readlines()
file.close()

f = []

alpha = np.deg2rad(0)  # x
beta = np.deg2rad(0)  # y
gamma = np.deg2rad(0)  # z

R = rot_matrix(alpha, beta, gamma)

v = []

for s in lines:
    sp = s.split()
    if not sp:
        continue
    if sp[0] == "v":
        raw_v = np.array([float(sp[1]), float(sp[2]), float(sp[3])], dtype=np.float64)
        v.append(R @ raw_v)

    if sp[0] == "f":
        f.append(
            [
                int(sp[1].split("/")[0]),
                int(sp[2].split("/")[0]),
                int(sp[3].split("/")[0]),
            ]
        )

v = np.array(v, dtype=np.float64)

xs = [p[0] for p in v]
ys = [p[1] for p in v]

cx = (min(xs) + max(xs)) / 2.0
cy = (min(ys) + max(ys)) / 2.0

tx = -cx
ty = -cy
tz = 0.0

t = np.array([tx, ty, tz], dtype=np.float64)
v = v + t

min_z = v[:, 2].min()
if min_z <= 1.0:
    dz = 1.0 - min_z
    v[:, 2] += dz

SCALE = 9000.0
u0, v0 = W / 2, H / 2

for k in range(len(f)):
    x0 = v[f[k][0] - 1][0]
    y0 = v[f[k][0] - 1][1]
    z0 = v[f[k][0] - 1][2]
    x1 = v[f[k][1] - 1][0]
    y1 = v[f[k][1] - 1][1]
    z1 = v[f[k][1] - 1][2]
    x2 = v[f[k][2] - 1][0]
    y2 = v[f[k][2] - 1][1]
    z2 = v[f[k][2] - 1][2]

    if z0 <= 0 or z1 <= 0 or z2 <= 0:
        continue

    x0S = SCALE * x0 / z0 + u0
    y0S = SCALE * y0 / z0 + v0
    x1S = SCALE * x1 / z1 + u0
    y1S = SCALE * y1 / z1 + v0
    x2S = SCALE * x2 / z2 + u0
    y2S = SCALE * y2 / z2 + v0

    normal = calc_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    nlen = np.linalg.norm(normal)
    if nlen == 0:
        continue
    cosin = np.dot(normal, [0, 0, 1]) / nlen

    if cosin < 0:
        shade = min(1.0, max(0.0, -cosin))
        draw_triangle(x0S, y0S, z0, x1S, y1S, z1, x2S, 
                      y2S, z2, zbuf, img_mat, (int(255 * shade), 0, 0),)

img = ImageOps.flip(Image.fromarray(img_mat, mode="RGB"))
img.save("img.png")