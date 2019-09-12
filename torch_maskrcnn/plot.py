import cv2
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv
import numpy as np
import math
plt.switch_backend('agg')
from numba import jit

def map_coco_to_personlab(keypoints):
    permute = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    return keypoints[:, permute, :]

def plot_poses(img, skeletons,boxes, save_name='pose.jpg'):
    EDGES = [
        (0, 14),
        (0, 13),
        (0, 4),
        (0, 1),
        (14, 16),
        (13, 15),
        (4, 10),
        (1, 7),
        (10, 11),
        (7, 8),
        (11, 12),
        (8, 9),
        (4, 5),
        (1, 2),
        (5, 6),
        (2, 3)
    ]
    NUM_EDGES = len(EDGES)
    
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    plt.figure()
    
    #img = img.astype('uint8')
    canvas = img.copy()
    for i in range(boxes.shape[0]):
        x0,y0,x1,y1=boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        cv.rectangle(canvas,(x0,y0),(x1,y1),(255,0,0),2)
    for i in range(17):
        rgba = np.array(cmap(1 - i/17. - 1./34))
        rgba[0:3] *= 255
        for j in range(len(skeletons)):
            cv.circle(canvas, tuple(skeletons[j][i, 0:2].astype('int32')), 2, colors[i], thickness=-1)

    to_plot = cv.addWeighted(img, 0.3, canvas, 0.7, 0)
    fig = matplotlib.pyplot.gcf()

    stickwidth = 2

    #skeletons = map_coco_to_personlab(skeletons)
    for i in range(NUM_EDGES):
        for j in range(len(skeletons)):
            edge = EDGES[i]
            if skeletons[j][edge[0],2] == 0 or skeletons[j][edge[1],2] == 0:
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            
    plt.imsave(save_name,canvas[:,:,:])
    plt.close()


def plot_poses2(img, skeletons, boxes, save_name='pose.jpg'):
    EDGES = [(0, 1), (1, 2), (2, 6), (6, 3), (3, 4), (4, 5),
             (10, 11), (11, 12), (12, 8), (8, 13), (13, 14), (14, 15),
             (6, 8), (8, 9)]
    NUM_EDGES = len(EDGES)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    #cmap = matplotlib.cm.get_cmap('hsv')
    plt.figure()

    # img = img.astype('uint8')
    canvas = img.copy()
    for i in range(boxes.shape[0]):
        x0, y0, x1, y1 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        cv.rectangle(canvas, (x0, y0), (x1, y1), (255, 0, 0), 2)
    for i in range(16):
        # rgba = np.array(cmap(1 - i / 16. - 1. / 32))
        # rgba[0:3] *= 255
        for j in range(len(skeletons)):
            cv.circle(canvas, tuple(skeletons[j][i, 0:2].astype('int32')), 2, colors[i], thickness=-1)

    # to_plot = cv.addWeighted(img, 0.3, canvas, 0.7, 0)
    # fig = matplotlib.pyplot.gcf()

    stickwidth = 2

    # skeletons = map_coco_to_personlab(skeletons)
    for i in range(NUM_EDGES):
        for j in range(len(skeletons)):
            edge = EDGES[i]
            if skeletons[j][edge[0], 2] == 0 or skeletons[j][edge[1], 2] == 0:
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv.fillConvexPoly(canvas, polygon, colors[i])
            #canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    plt.imsave(save_name, canvas[:, :, :])
    plt.close()

def plot_poses3(img, points, boxes, save_name='pose.jpg'):
    edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
             [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
             [6, 8], [8, 9]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    c=[255,255,0]
    canvas = img.copy()

    for i in range(boxes.shape[0]):
        x0, y0, x1, y1 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        cv.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
    for i in range(16):
        for j in range(len(points)):
            cv.circle(img, tuple(points[j][i, 0:2].astype('int32')), 2, colors[i], thickness=-1)


    for i in range (len(edges)):
        for j in range(len(points)):
            position = give_pixel(edges[i], points[j])

            cv2.line(img, tuple(position[0])[0:2], tuple(position[1])[0:2], c, 2)

    cv2.imwrite(save_name,img)
    # plt.imsave(save_name, img)
    # plt.close()

def give_pixel(link,joints):

    return (joints[link[0]].astype(np.int), joints[link[1]].astype(np.int))








class Plot_save3d(object):

    def __init__(self, edges,ipynb=False ):
        self.ipynb = ipynb
        if not self.ipynb:
            self.plt = plt
            self.fig = self.plt.figure()
            self.ax = self.fig.add_subplot(111,projection='3d')
            self.ax.set_xlabel('x')
            self.ax.set_ylabel('z')
            self.ax.set_zlabel('y')
            self.ax.grid(False)
        oo = 1e10
        self.xmax, self.ymax, self.zmax = -oo, -oo, -oo
        self.xmin, self.ymin, self.zmin = oo, oo, oo
        self.imgs = {}
        self.edges = edges

    def add_point_3d(self, points, save_name,c='b', marker='o', edges=None):
        if edges == None:
            edges = self.edges
        # show3D(self.ax, point, c, marker = marker, edges)
        #points = points.reshape(-1, 3)
        x, y, z = np.zeros((3, points.shape[0]))
        for j in range(points.shape[0]):
            x[j] = points[j, 0].copy()
            y[j] = points[j, 2].copy()#*10-100
            z[j] = - points[j, 1].copy()
            self.xmax = max(x[j], self.xmax)
            self.ymax = max(y[j], self.ymax)
            self.zmax = max(z[j], self.zmax)
            self.xmin = min(x[j], self.xmin)
            self.ymin = min(y[j], self.ymin)
            self.zmin = min(z[j], self.zmin)
        if c == 'auto':
            c = [(z[j] + 0.5, y[j] + 0.5, x[j] + 0.5) for j in range(points.shape[0])]
        self.ax.scatter(x, y, z, s=60, c=c, marker=marker)
        self.ax.scatter(np.array([0]), np.array([0]), np.array([0]), s=100, c='r', marker=marker)
        #print(points)
        for e in edges:
            #print(x.shape,y.shape,z.shape,e)
            self.ax.plot(x[e], y[e], z[e], c=c)
        self.plt.show()
        self.plt.savefig(save_name)
    def plot3d(self,points,save_name):
        for i in range(len(points)):
            self.add_point_3d(points[i],save_name)
