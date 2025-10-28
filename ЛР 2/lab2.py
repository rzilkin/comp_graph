import numpy as np
from PIL import Image, ImageOps


def calc_barCord(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / (
        (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    )
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / (
        (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    )
    lambda2 = 1.0 - lambda0 - lambda1

    return (lambda0, lambda1, lambda2)


def calc_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    return np.cross((x1 - x2, y1 - y2, z1 - z2), (x1 - x0, y1 - y0, z1 - z0))


def draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, z_buf, matrix, color):
    xmin = int(min(x0, x1, x2))
    xmax = int(max(x0, x1, x2))
    ymin = int(min(y0, y1, y2))
    ymax = int(max(y0, y1, y2))

    xmax = min(xmax, matrix.shape[1] - 1)
    ymax = min(ymax, matrix.shape[0] - 1)

    if xmin > xmax or ymin > ymax:
        return

    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0

    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            pixel = calc_barCord(x, y, x0, y0, x1, y1, x2, y2)
            if pixel[0] >= 0 and pixel[1] >= 0 and pixel[2] >= 0:
                z_hat = pixel[0] * z0 + pixel[1] * z1 + pixel[2] * z2
                if z_hat < z_buf[y, x]:
                    z_buf[y, x] = z_hat
                    matrix[y, x] = color


img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)
zbuf = np.full((2000, 2000), np.inf, dtype=np.float64)


file = open("model.obj")
f = []
v = []

for s in file:
    sp = s.split()
    if sp[0] == "v":
        v.append([float(sp[1]), float(sp[2]), float(sp[3])])

    if sp[0] == "f":
        f.append(
            [
                int(sp[1].split("/")[0]),
                int(sp[2].split("/")[0]),
                int(sp[3].split("/")[0]),
            ]
        )


for k in range(len(f)):
    x0 = v[f[k][0] - 1][0] * 9000 + 1000
    y0 = v[f[k][0] - 1][1] * 9000 + 1000
    z0 = v[f[k][0] - 1][2] * 9000
    x1 = v[f[k][1] - 1][0] * 9000 + 1000
    y1 = v[f[k][1] - 1][1] * 9000 + 1000
    z1 = v[f[k][1] - 1][2] * 9000
    x2 = v[f[k][2] - 1][0] * 9000 + 1000
    y2 = v[f[k][2] - 1][1] * 9000 + 1000
    z2 = v[f[k][2] - 1][2] * 9000

    normal = calc_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    cosin = (
        np.dot(normal, [0, 0, 1]) / np.linalg.norm(normal) * np.linalg.norm([0, 0, 1])
    )

    if cosin < 0:
        draw_triangle(
            x0, y0, z0, x1, y1, z1, x2, y2, z2, zbuf, img_mat, (-255 * cosin, 0, 0)
        )


img = ImageOps.flip(Image.fromarray(img_mat, mode="RGB"))

img.save("img.png")
