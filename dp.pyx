#%%cython -f -c-fopenmp --link-args=-fopenmp -c-g
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
import numpy as np 
cimport numpy as np 
from cython.parallel import prange

cdef int size = 5
cdef inline float amin(float a,float b) nogil:
    if a<b :
        return a
    else :
        return b

cdef transfer(int i,int j,int x,int y,int dis_num,int dic,float p1,float p2,np.ndarray[np.float64_t, ndim=4] lost,np.ndarray[np.float64_t, ndim=3] ms ):               
    
    cdef Py_ssize_t d1,d2
    cdef float min_flag 
    for d1 in prange(dis_num,nogil=True,schedule='guided',num_threads=size,chunksize=10):
    #for d1 in range(dis_num):
        min_flag = 10000
        for d2 in range(dis_num):
            if d1-d2 == 0:
                min_flag = amin(min_flag,lost[dic,x,y,d2])
            elif d1 - d2 == 1 or d2 - d1 == 1:
                min_flag = amin(min_flag,lost[dic,x,y,d2]+p1)
            else:
                min_flag = amin(min_flag,lost[dic,x,y,d2]+p2)
        lost[dic,i,j,d1] = -ms[i,j,d1] + min_flag

cpdef  dp_mrf(np.ndarray[np.float64_t, ndim=3] ms,float p1,float p2,int dis_num):
    
    cdef int ymax = ms.shape[0],xmax = ms.shape[1]
    cdef Py_ssize_t i,j,d
    cdef np.ndarray[np.float64_t, ndim=4] lost = np.zeros([8,ymax,xmax,dis_num], dtype = float)
     
    for i in range(ymax):
        for d in prange(dis_num,nogil=True,schedule='guided',num_threads=size,chunksize=10):
            lost[0,i,0,d] = -ms[i,0,d]
            lost[1,i,xmax-1,d] = -ms[i,xmax-1,d]

        for j in range(1,xmax):
            transfer(i,j,i,j-1,dis_num,0,p1,p2,lost,ms)     

        for j in range(xmax-2,-1,-1):
            transfer(i,j,i,j+1,dis_num,1,p1,p2,lost,ms)

    for j in range(xmax):
        
        for d in prange(dis_num,nogil=True,schedule='guided',num_threads=size,chunksize=10):
            lost[2,0,j,d] = -ms[0,j,d]
            lost[3,ymax-1,j,d] = -ms[ymax-1,j,d]

        for i in range(1,ymax):
            transfer(i,j,i-1,j,dis_num,2,p1,p2,lost,ms)

        for i in range(ymax-2,-1,-1):
            transfer(i,j,i+1,j,dis_num,3,p1,p2,lost,ms)

    for j in range(4,8):
        for i in range(ymax):
            for d in prange(dis_num,nogil=True,schedule='guided',num_threads=size,chunksize=10):
                lost[j,i,0,d] = -ms[i,0,d]
                lost[j,i,xmax-1,d] = -ms[i,xmax-1,d]
        for i in range(xmax):
            for d in prange(dis_num,nogil=True,schedule='guided',num_threads=size,chunksize=10):
                lost[j,0,i,d] = -ms[0,i,d]
                lost[j,ymax-1,i,d] = -ms[ymax-1,i,d]

    for i in range(1,ymax):
        for j in range(1,xmax):
            transfer(i,j,i-1,j-1,dis_num,4,p1,p2,lost,ms)
        for j in range(xmax-2,-1,-1):
            transfer(i,j,i-1,j+1,dis_num,5,p1,p2,lost,ms)

    for i in range(ymax-2,-1,-1):
        for j in range(1,xmax):
            transfer(i,j,i+1,j-1,dis_num,6,p1,p2,lost,ms)
        for j in range(xmax-2,-1,-1):
            transfer(i,j,i+1,j+1,dis_num,7,p1,p2,lost,ms)

    return lost