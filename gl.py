#Escritor de Imagenes BMP
#Author: Diego Estrada

import struct
import random
from obj import Obj
from collections import namedtuple

#math utils

V2 = namedtuple('Point2', ['x', 'y'])
V3 = namedtuple('Point3', ['x', 'y', 'z'])

def sum(v0, v1):
  return V3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z)

def sub(v0, v1):
  return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

def mul(v0, k):
  return V3(v0.x * k, v0.y * k, v0.z *k)

def dot(v0, v1):
  return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

def cross(v0, v1):
  return V3(
    v0.y * v1.z - v0.z * v1.y,
    v0.z * v1.x - v0.x * v1.z,
    v0.x * v1.y - v0.y * v1.x,
  )

def length(v0):
  return (v0.x**2 + v0.y**2 + v0.z**2)**0.5

def norm(v0):
  v0length = length(v0)

  if not v0length:
    return V3(0, 0, 0)

  return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)

def bbox(*vertices):
  xs = [ vertex.x for vertex in vertices ]
  ys = [ vertex.y for vertex in vertices ]
  xs.sort()
  ys.sort()

  return V2(xs[0], ys[0]), V2(xs[-1], ys[-1])

def barycentric(A, B, C, P):
  
  bary = cross(
    V3(C.x - A.x, B.x - A.x, A.x - P.x), 
    V3(C.y - A.y, B.y - A.y, A.y - P.y)
  )

  if abs(bary[2]) < 1:
    return -1, -1, -1   

  return (
    1 - (bary[0] + bary[1]) / bary[2], 
    bary[1] / bary[2], 
    bary[0] / bary[2]
  )

#bmp utils

def char(c):
    return struct.pack('=c', c.encode('ascii'))

def word(c):
    return struct.pack('=h', c)

def dword(c):
    return struct.pack('=l', c)

def color(r, g, b):
    return bytes([b,g,r])

#My Renderer

class Render(object):
    def __init__(self):
        self.clear_color = color(0,0,0)
        self.draw_color = color(255,255,233)
    
    def glClear(self):
        self.framebuffer = [
            [self.clear_color for x in range(self.width)] 
            for y in range(self.height)
        ]
        self.zbuffer = [
            [-float('inf') for x in range(self.width)]
            for y in range(self.height)
        ]

    def glCreateWindow(self, width, height): #el width y height del window es el del Render()
        self.width = width
        self.height = height
        self.framebuffer = []
        self.glClear()
    
    def point(self, x,y, color=None):
        self.framebuffer[y][x] = color or self.draw_color

    def glInit(self):
        pass

    def glViewPort(self, x, y, width, height):
        self.x_VP = x
        self.y_VP = y
        self.width_VP = width
        self.height_VP = height

    def glClearColor(self, r, g, b):
        self.clear_color = color(int(round(r*255)),int(round(g*255)),int(round(b*255)))

    def glColor(self, r,g,b):
        self.draw_color = color(int(round(r*255)),int(round(g*255)),int(round(b*255)))

    def glVertex(self, x, y):
        xPixel = round((x+1)*(self.width_VP/2)+self.x_VP)
        yPixel = round((y+1)*(self.height_VP/2)+self.y_VP)
        self.point(xPixel, yPixel)
    
    def glLine(self,x1, y1, x2, y2):
        x1 = int(round((x1+1) * self.width / 2))
        y1 = int(round((y1+1) * self.height / 2))
        x2 = int(round((x2+1) * self.width / 2))
        y2 = int(round((y2+1) * self.height / 2))
        steep=abs(y2 - y1)>abs(x2 - x1)
        if steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        if x1>x2:
            x1,x2 = x2,x1
            y1,y2 = y2,y1

        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        y = y1
        offset = 0
        threshold = dx

        for x in range(x1, x2):
            if offset>=threshold:
                y += 1 if y1 < y2 else -1
                threshold += 2*dx
            if steep:
                self.framebuffer[x][y] = self.draw_color
            else:
                self.framebuffer[y][x] = self.draw_color
            offset += 2*dy

    def load(self, filename, translate=(0, 0, 0), scale=(1, 1, 1)):
        model = Obj(filename)

        light = V3(0,0,1)

        for face in model.vfaces:
            vcount = len(face)

            if vcount == 3:
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1

                a = self.transform(model.vertices[f1], translate, scale)
                b = self.transform(model.vertices[f2], translate, scale)
                c = self.transform(model.vertices[f3], translate, scale)

                normal = norm(cross(sub(b, a), sub(c, a)))
                intensity = dot(normal, light)
                grey = round(255 * intensity)
                if grey < 0:
                    continue  
            
                self.triangle(a, b, c, color(grey, grey, grey))
            else:
                # assuming 4
                f1 = face[0][0] - 1
                f2 = face[1][0] - 1
                f3 = face[2][0] - 1
                f4 = face[3][0] - 1   

                vertices = [
                    self.transform(model.vertices[f1], translate, scale),
                    self.transform(model.vertices[f2], translate, scale),
                    self.transform(model.vertices[f3], translate, scale),
                    self.transform(model.vertices[f4], translate, scale)
                ]

                normal = norm(cross(sub(vertices[0], vertices[1]), sub(vertices[1], vertices[2])))  # no necesitamos dos normales!!
                intensity = dot(normal, light)
                grey = round(255 * intensity)
                if grey < 0:
                    continue 
        
                A, B, C, D = vertices 
                
                self.triangle(A, B, C, color(grey, grey, grey))
                self.triangle(A, C, D, color(grey, grey, grey))

    def glFinish(self, filename):
        f = open(filename, 'bw')

        #file header
        f.write(char('B'))
        f.write(char('M'))
        f.write(dword(14 + 40 + self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(14 + 40))

        #image header
        f.write(dword(40))
        f.write(dword(self.width))
        f.write(dword(self.height))
        f.write(word(1))
        f.write(word(24))   
        f.write(dword(0))
        f.write(dword(24))
        f.write(dword(self.width * self.height * 3))
        f.write(dword(0))
        f.write(dword(0))
        f.write(dword(0)) 
        f.write(dword(0))

        # pixel data

        for x in range(self.width):
            for y in range(self.height):
                f.write(self.framebuffer[x][y])
        
        f.close()
    
    def glFill(self, polygon):
        for y in range(self.height):
            for x in range(self.width):
                i = 0
                j = len(polygon) - 1
                inside = False
                for i in range(len(polygon)):
                    if (polygon[i][1] < y and polygon[j][1] >= y) or (polygon[j][1] < y and polygon[i][1] >= y):
                        if polygon[i][0] + (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) * (polygon[j][0] - polygon[i][0]) < x:
                            inside = not inside
                    j = i
                if inside:
                    self.point(y,x)
    def shader(self, x,y,z):
      if z>245:
        return color(214,252,255)
      elif z>244:
        return color(211,249,252)
      elif z>243:
        return color(208,246,249)
      elif z>240:
        return color(205,243,246)
      elif z>235:
        return color(199,237,240)
      elif z>230:
        return color(193,231,234)
      elif z>220:
        return color(187,225,228)
      elif z>210:
        return color(181,219,222)
      elif z>200:
        return color(175,213,216)
      elif z>190:
        return color(169,207,210)
      elif z>180:
        return color(163,201,204)
      elif z>170:
        return color(157,195,198)
      elif z>160:
        return color(151,189,192)
      elif z>150:
        return color(145,173,176)
      elif z>140:
        return color(139,167,170)
      elif z>130:
        return color(130,158,161) #9
      elif z>120:
        return color(124,152,155)
      elif z>110:
        return color(112,140,143)
      elif z>90:
        return color(106,134,137)
      elif z>70:
        return color(100,128,131)
      elif z>50:
        return color(94,122,125)
      elif z>30:
        return color(88,116,119)
      elif z>20:
        return color(82,110,113)
      else:
        return color(76,104,107)

    def triangle(self, A, B, C, color):
        bbox_min, bbox_max = bbox(A, B, C)

        for x in range(bbox_min.x, bbox_max.x + 1):
            for y in range(bbox_min.y, bbox_max.y + 1):
                w, v, u = barycentric(A, B, C, V2(x, y))
                if w < 0 or v < 0 or u < 0:  
                    continue
                
                z = A.z * w + B.z * v + C.z * u
                color = self.shader(x,y,z)
                if z > self.zbuffer[x][y]:
                    self.point(x, y,color)
                    self.zbuffer[x][y] = z

    def transform(self, vertex, translate=(0, 0, 0), scale=(1, 1, 1)):
    
        return V3(
        round((vertex[0] + translate[0]) * scale[0]),
        round((vertex[1] + translate[1]) * scale[1]),
        round((vertex[2] + translate[2]) * scale[2])
        )

r = Render()
r.glCreateWindow(1000,1000)
r.load('./sphere.obj', (1, 1, 0), (500, 500, 500))
r.glFinish('out.bmp')
