import numpy as np
import os
from skimage.segmentation import find_boundaries
from matplotlib import pyplot as plt

from scipy import spatial


def IoU(a1,a2):
    smooth = 2e-12
    intersection = np.sum(np.logical_and(a1!=0,a2!=0))
    union = np.sum(np.logical_or(a1!=0,a2!=0))
    
    return intersection/(union+smooth)
    
    

def sortByDistance(point_list):
    
    point_list = np.array(point_list)
    
    
    new_point_list = []
    
    new_point_list.append(point_list[0,:])
    #remove that point from the point list
    point_list = np.delete(point_list,0,axis=0)
    
    for x in range(0,len(point_list)):
        pt = new_point_list[-1]
        _,index = spatial.KDTree(point_list).query(pt)
        new_point_list.append(point_list[index,:])
        #remove that point from the point list
        point_list = np.delete(point_list,index,axis=0)
        
        
    return  np.array(new_point_list)


def sampleContour(contour,nb_points=25):
    
    loc = np.nonzero(contour)
    
    nb = len(loc[0]) #number of points of the boundary
      
    idx = np.linspace(0,nb,nb_points,endpoint=False)
    idx = [int(i) for i in idx]
    
    if nb > 0:

        points = [[a,b] for a,b in zip(loc[0],loc[1])]
        points = sortByDistance(points)   
        if nb == nb_points:
            return points
        elif nb < nb_points:
            #add redudant points
            diff = int(np.ceil(nb_points/nb)) 
            points = np.repeat(points,diff+1,axis=0)

                
        points = points[idx,...]
        
        return points
    else:
        return []  



def normalizeHU(npzarray,maxHU=400.,minHU=-1200.):
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def normFunc(im):

    if np.mean(im) < -300:
        pass
    else:
        im = im - 1024
        
    im = normalizeHU(im)

    return np.float64(im)

def minmaxNorm(a):
    return (a-np.min(a))/(np.max(a)-np.min(a))
    
def normFunc_simple(im):

    if np.mean(im) < -300:
        pass
    else:
        im = im - 1024
    
    
    im[im<-1200] = -1200
    im[im>400] = 400
    
    return np.float64(im)
    
def findExtension(directory,extension='.npy'):
    files = []
    full_path = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            files += [file]
            full_path += [os.path.join(directory,file)]
            
    files.sort()
    full_path.sort()
    return files, full_path
    
def gravFields(centers,M=100,dim=64,normalization=None):
    # centers is a list of tuples containing the coordinates of each dark hole center
    # the origin of the reference is the upper left corner
    #x are columns and y are rows
    
    C = []

    for  c in centers:
        C_ = [cc-dim/2 for cc in c]
        C.append(C_)
    

    amin = -dim/2
    amax = dim/2        
    x, y, z = np.meshgrid(np.arange(amin, amax), np.arange(amin, amax), np.arange(amin, amax))
    
    out = np.zeros((dim,dim,dim))
      
    for c in C:
        x__ = x-c[0]
        y__ = y-c[1]
        z__ = z-c[2]
        temp = x__**2 + y__**2 + z__**2
        temp += M
        
        out += 1/(temp+2e-12)
        out += temp

    
    if normalization:
        out = normFunc(out)
    
    return out
def computeField3D(amin,amax,center,weight_mode='constant',p=0):
    
    x, y, z = np.meshgrid(np.arange(amin, amax), np.arange(amin, amax), np.arange(amin, amax))
    
    x_ = x- center[0]
    y_ = y- center[1]
    z_ = z- center[2]
    norm = np.sqrt((2*x_)**2+(2*y_)**2+(2*z_)**2)

    radius = np.copy(norm)
    
    x_ /= norm
    y_ /= norm
    z_ /= norm
     
  
    if weight_mode == 'linear':        
        x_ /= radius
        y_ /= radius
        z_ /= radius    
    if weight_mode == 'sqrt':        
        x_ /= np.sqrt(radius)
        y_ /= np.sqrt(radius)
        z_ /= np.sqrt(radius)
    if weight_mode == 'free':
        x_ /= radius**p
        y_ /= radius**p
        z_ /= radius**p            
    
#
#    plt.figure()
#    plt.imshow(norm[...,32])    
    
    x_ = np.nan_to_num(x_)
    y_ = np.nan_to_num(y_)
    z_ = np.nan_to_num(z_)
        
    return x_, y_, z_   
    
def attractionField3D(center_in, center_out,dim=(64,64,64),normalization=minmaxNorm,weight_mode='constant',p=0):
    # centers is a list of tuples containing the coordinates of each dark hole center
    # the origin of the reference is the upper left corner
    #x are columns and y are rows
    
    in_ = []
    out_ = []
    
    for c in center_in:
        in_.append([cc-dim[i]/2 for i,cc in enumerate(c)])
    for c in center_out:
        out_.append([cc-dim[i]/2 for i,cc in enumerate(c)])

    amin = -dim[0]/2
    amax = dim[1]/2        
    
        
    out_x = np.zeros(dim)
    out_y = np.zeros(dim)
    out_z = np.zeros(dim)
    
    
    
    for i,c in enumerate(in_):
        x_,y_,z_ = computeField3D(amin,amax,c,weight_mode=weight_mode,p=p)
        out_x += x_
        out_y += y_
        out_z += z_
    for i,c in enumerate(out_):
        x_,y_,z_= computeField3D(amin,amax,c,weight_mode=weight_mode,p=p)
        out_x += x_*-1
        out_y += y_*-1      
        out_z += z_*-1
    
    mag = np.sqrt(out_x**2+out_y**2+out_z**2)
    
    if normalization and np.max(mag)!=np.min(mag):
        mag = normalization(mag)
    
    
    return out_x,out_y, out_z  ,mag  

def findDistantPoints2D(seg,iterations=100,compute_boundary=True):
    #go to that slice and get the boundary
    if compute_boundary:
        bound = find_boundaries(seg!=0,connectivity=2,mode='inside')
        bound = bound!=0
    else:
        bound = seg
    
    #collect a random point from the boundary
    bound_p = np.nonzero(bound)
    
    idx = np.array([x for x in range(len(bound_p[0]))])
    
    dist = 0
    best = []
    
    
    for x in range(0,iterations):
        np.random.shuffle(idx) 
        p1 = [bound_p[1][idx[0]],bound_p[0][idx[0]]]
        p2 = [bound_p[1][idx[1]],bound_p[0][idx[1]]]
        
        curr_distance = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        if curr_distance > dist:
            dist = curr_distance
            best = [p1,p2]
    return best  
    
def generateCirclePoints(radius,nb_points,size=64):
    amin = -size/2
    amax = size/2        
    x, y = np.meshgrid(np.arange(amin, amax), np.arange(amin, amax))
    
    circle = (x**2+y**2)<=radius**2
    contour = find_boundaries(circle,mode='inner') #inner
    points = sampleContour(contour,nb_points=nb_points)    
    points = sortByDistance(points)
    return points