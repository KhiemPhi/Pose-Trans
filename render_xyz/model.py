'''
This code is copied from
https://github.com/wadimkehl/ssd-6d.git
'''

import os
import numpy as np
from scipy.spatial.distance import pdist
from plyfile import PlyData
import cv2
from vispy import gloo

class Model3D:
    def __init__(self, file_to_load=None):
        self.vertices = None
        self.centroid = None
        self.indices = None
        self.colors = None
        self.texcoord = None
        self.texture = None
        self.collated = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.bb = None
        self.bb_vbuffer = None
        self.bb_ibuffer = None
        self.diameter = None
        if file_to_load:
            self.load(file_to_load)

    def _compute_bbox(self,color_type=0):

        self.bb = []
        minx, maxx = min(self.vertices[:, 0]), max(self.vertices[:, 0])
        miny, maxy = min(self.vertices[:, 1]), max(self.vertices[:, 1])
        minz, maxz = min(self.vertices[:, 2]), max(self.vertices[:, 2])
        self.bb.append([minx, miny, minz])
        self.bb.append([minx, maxy, minz])
        self.bb.append([minx, miny, maxz])
        self.bb.append([minx, maxy, maxz])
        self.bb.append([maxx, miny, minz])
        self.bb.append([maxx, maxy, minz])
        self.bb.append([maxx, miny, maxz])
        self.bb.append([maxx, maxy, maxz])
        self.bb = np.asarray(self.bb, dtype=np.float32)
        #self.diameter = max(pdist(self.bb, 'euclidean'))

        # Set up rendering data
        if(color_type==0):
            colors = [[1, 0, 0],[1, 1, 0], [0, 1, 0], [0, 1, 1],
                      [0, 0, 1], [0, 1, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        elif(color_type==1):
            colors = [[0, 0, 1],[0, 0, 1], [0, 0, 1], [0, 0, 1],
                      [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
        elif(color_type==2):
            colors = [[0, 1, 0],[0, 1, 0], [0, 1, 0], [0, 1, 0],
                      [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
        elif(color_type==3):
            colors = [[1, 1, 1],[1, 1, 1], [1, 1, 1], [1, 1, 1],
                      [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        else:
            colors = [[0, 1,0],[0, 1, 0], [0, 1, 0], [0, 1, 0],
                      [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]

        indices = [0, 1, 0, 2, 3, 1, 3, 2,
                   4, 5, 4, 6, 7, 5, 7, 6,
                   0, 4, 1, 5, 2, 6, 3, 7]

        vertices_type = [('a_position', np.float32, 3), ('a_color', np.float32, 3)]
        collated = np.asarray(list(zip(self.bb, colors)), vertices_type)
        self.bb_vbuffer = gloo.VertexBuffer(collated)
        self.bb_ibuffer = gloo.IndexBuffer(indices)

    def load(self, path, demean=False, scale=1.0, colored=True):
        # param::colored: whether this is the colored model by tools/2_1_ply_file_to_3d_coord_model.py
        data = PlyData.read(path)
        self.vertices = np.zeros((data['vertex'].count, 3))
        self.vertices[:, 0] = np.array(data['vertex']['x'])
        self.vertices[:, 1] = np.array(data['vertex']['y'])
        self.vertices[:, 2] = np.array(data['vertex']['z'])
        self.vertices *= scale
        self.centroid = np.mean(self.vertices, 0)

        if demean:
            self.centroid = np.zeros((1, 3), np.float32)
            self.vertices -= self.centroid

        self._compute_bbox()

        self.indices = np.asarray(list(data['face']['vertex_indices']), np.uint32)

        # Look for texture map as jpg or png
        filename = path.split('/')[-1]
        abs_path = path[:path.find(filename)]
        tex_to_load = None
        if os.path.exists(abs_path + filename[:-4] + '.jpg'):
            tex_to_load = abs_path + filename[:-4] + '.jpg'
        elif os.path.exists(abs_path + filename[:-4] + '.png'):
            tex_to_load = abs_path + filename[:-4] + '.png'

        # Try to read out texture coordinates
        if tex_to_load is not None:
            print('Loading {} with texture {}'.format(filename, tex_to_load))
            image = cv2.flip(cv2.imread(tex_to_load, cv2.IMREAD_UNCHANGED), 0)  # Must be flipped because of OpenGL
            self.texture = gloo.Texture2D(image)

            # If texcoords are face-wise
            if 'texcoord' in str(data):
                self.texcoord = np.asarray(list(data['face']['texcoord']))
                assert self.indices.shape[0] == self.texcoord.shape[0]  # Check same face count
                temp = np.zeros((data['vertex'].count, 2))
                temp[self.indices.flatten()] = self.texcoord.reshape((-1, 2))
                self.texcoord = temp

            # If texcoords are vertex-wise
            elif 'texture_u' in str(data):
                self.texcoord = np.zeros((data['vertex'].count, 2))
                self.texcoord[:, 0] = np.array(data['vertex']['texture_u'])
                self.texcoord[:, 1] = np.array(data['vertex']['texture_v'])

        # If texture coords loaded succesfully
        if self.texcoord is not None:
            vertices_type = [('a_position', np.float32, 3), ('a_texcoord', np.float32, 2)]
            self.collated = np.asarray(list(zip(self.vertices, self.texcoord)), vertices_type)

        # Otherwise fall back to vertex colors
        else:
            self.colors = 0.5*np.ones((data['vertex'].count, 3))
            if not colored:
                # use the 3D coordinate as the color
                print('Loading {} with vertex coordinate as colors'.format(filename))
                self.colors = self.vertices.copy()
                # but need to normalize in [0, 1] to render in opengl
                print('color min {} max {}'.format(self.colors.min(), self.colors.max()))
                self.raw_color_min = self.colors.min()
                self.raw_color_extent = self.colors.max() - self.raw_color_min
                self.colors = (self.colors - self.raw_color_min) / self.raw_color_extent
                print('color min {} max {}'.format(self.colors.min(), self.colors.max()))  # check whether the color is in [0, 1]
            elif 'blue' in str(data):
                print('Loading {} with vertex colors'.format(filename))
                self.colors[:, 0] = np.array(data['vertex']['blue'])
                self.colors[:, 1] = np.array(data['vertex']['green'])
                self.colors[:, 2] = np.array(data['vertex']['red'])
                self.colors /= 255.0
            else:
                print('Loading {} without any coloring!!'.format(filename))
            vertices_type = [('a_position', np.float32, 3), ('a_color', np.float32, 3)]
            self.collated = np.asarray(list(zip(self.vertices, self.colors)), vertices_type)

        self.vertex_buffer = gloo.VertexBuffer(self.collated)
        self.index_buffer = gloo.IndexBuffer(self.indices.flatten())

# todo: handle texture_paths not None?
# todo: support cache
def load_models(model_paths, obj_ids, vertex_scale=0.001, center=False):
  
    models_all = {}
    for m_id, m_path in enumerate(model_paths):
        m = Model3D()
        m.load(m_path, demean=center, scale=vertex_scale, colored=True)
        models_all[obj_ids[m_id+1]] = m
    return models_all
