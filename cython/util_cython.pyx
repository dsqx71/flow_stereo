#!python
# cython: boundscheck=False , wraparound=False, overflowcheck=False,
# optimize.use_switch=True, optimize.unpack_method_calls=True,
import cython
import numpy as np
cimport numpy as np
from libc.math cimport ceil,floor,round

cdef inline double double_max(double a,double b):

    if a>b:
        return a
    else:
        return b

cpdef resize(np.ndarray[np.float64_t, ndim=2] src, int x_shape, int y_shape, float threshold=0.5):
    """
    This function is different from opencv's counterpart! it will consider NaN_point ratio.
    Parameters
    ----------
    src : array with np.float64
        scr only has two axis
    x_shape : int
        target width
    y_shape : int
        target height
    threshold : float
        indicate whether the point should be filled.
    Returns
    -------
    dst : ndarray
        resized image
    """

    cdef double y_ori = src.shape[0],x_ori = src.shape[1]
    cdef Py_ssize_t i,j,desty,destx,yoff,xoff
    cdef double factory = (y_ori-1) / float(y_shape-1) ,factorx = (x_ori-1) / float(x_shape-1)
    cdef double accum_value,accum_weight,accum_nan,sample,weight,botx,boty
    cdef np.ndarray[np.float64_t, ndim=2] dst = np.zeros([y_shape,x_shape], dtype = float)

    cdef  double y,x,w1,w2,w3,w4,tot,tmp1,tmp2
    cdef  int ibotx,iboty, wradius = int(ceil(factorx)), hradius = int(ceil(factory)),cx,cy

    if y_ori == y_shape and x_ori==x_shape:
        return src

    for desty in range(y_shape):
        for destx in range(x_shape):

            accum_value = 0
            accum_weight = 0
            accum_nan = 0

            botx =  float(destx)/float(x_shape-1)*float(x_ori-1)
            boty =  float(desty)/float(y_shape-1)*float(y_ori-1)

            ibotx = int(round(botx))
            iboty = int(round(boty))

            for yoff in range(-hradius,hradius+1):
                for xoff in range(-wradius,wradius+1):

                    cy = iboty + yoff
                    cx = ibotx + xoff

                    if cx>=0 and cx<x_ori and cy>=0 and cy<y_ori:

                        sample = float(src[cy,cx])
                        tmp1 = 1-(float(abs(cx - botx))/ factorx)
                        tmp2 = 1.0- float((abs(cy - boty))/factory)
                        weight = double_max(0.0,tmp1) * double_max(0.0,tmp2)

                        if(sample != sample):
                            # sample != sample means value of the point is NaN
                            accum_nan += weight
                            sample = 0
                            weight = 0

                        accum_value += sample * weight
                        accum_weight += weight

            if accum_nan / (accum_weight+1e-14) > threshold:
                dst[desty,destx] = np.nan
            else:
                dst[desty,destx] = accum_value / (accum_weight+1e-5)
    return dst
