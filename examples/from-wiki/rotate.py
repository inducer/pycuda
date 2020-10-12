#!python 
#!/usr/bin/env python -tt
# encoding: utf-8
#
# Created by Holger Rapp on 2009-03-11.
# HolgerRapp@gmx.net
#

import pycuda.driver as cuda
import pycuda.compiler
import pycuda.autoinit
import numpy
from math import pi,cos,sin

_rotation_kernel_source = """
texture<float, 2> tex;

__global__ void copy_texture_kernel(
    const float resize_val, 
    const float alpha, 
    unsigned short oldiw, unsigned short oldih,
    unsigned short newiw, unsigned short newih,
    unsigned char* data) {

        // calculate pixel idx
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        // We might be outside the reachable pixels. Don't do anything
        if( (x >= newiw) || (y >= newih) )
            return;
        
        // calculate offset into destination array
        unsigned int didx = y * newiw + x;
        
        // calculate offset into source array (be aware of rotation and scaling)
        float xmiddle = (x-newiw/2.) / resize_val;
        float ymiddle = (y-newih/2.) / resize_val;
        float sx = ( xmiddle*cos(alpha)+ymiddle*sin(alpha) + oldiw/2.) ;
        float sy = ( -xmiddle*sin(alpha)+ymiddle*cos(alpha) + oldih/2.);
        
        if( (sx < 0) || (sx >= oldiw) || (sy < 0) || (sy >= oldih) ) { 
            data[didx] = 255; 
            return;
        }

        data[didx] = tex2D(tex, sx, sy);
    }
"""
mod_copy_texture=pycuda.compiler.SourceModule( _rotation_kernel_source )

copy_texture_func = mod_copy_texture.get_function("copy_texture_kernel")
texref = mod_copy_texture.get_texref("tex")

def rotate_image( a, resize = 1.5, angle = 20., interpolation = "linear", blocks = (16,16,1)  ):
    """
    Rotates the array. The new array has the new size and centers the
    picture in the middle.
    
    a             - array (2-dim)
    resize        - new_image w/old_image w
    angle         - degrees to rotate the image
    interpolation - "linear" or None
    blocks        - given to the kernel when run


    returns: a new array with dtype=uint8 containing the rotated image
    """
    angle = angle/180. *pi
    
    # Convert this image to float. Unsigned int texture gave 
    # strange results for me. This conversion is slow though :(
    a = a.astype("float32")

    # Calculate the dimensions of the new image
    calc_x = lambda x_y: (x_y[0]*a.shape[1]/2.*cos(angle)-x_y[1]*a.shape[0]/2.*sin(angle))
    calc_y = lambda x_y1: (x_y1[0]*a.shape[1]/2.*sin(angle)+x_y1[1]*a.shape[0]/2.*cos(angle))

    xs = [ calc_x(p) for p in [ (-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.) ] ]
    ys = [ calc_y(p) for p in [ (-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.) ] ]

    new_image_dim = (
        int(numpy.ceil(max(ys)-min(ys))*resize),
        int(numpy.ceil(max(xs)-min(xs))*resize),
    )
    
    # Now generate the cuda texture
    cuda.matrix_to_texref(a, texref, order="C")
    
    # We could set the next if we wanted to address the image
    # in normalized coordinates ( 0 <= coordinate < 1.)
    # texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
    if interpolation == "linear":
        texref.set_filter_mode(cuda.filter_mode.LINEAR)

    # Calculate the gridsize. This is entirely given by the size of our image. 
    gridx = new_image_dim[0]/blocks[0] if \
            new_image_dim[0]%blocks[0]==1 else new_image_dim[0]/blocks[0] +1
    gridy = new_image_dim[1]/blocks[1] if \
            new_image_dim[1]%blocks[1]==0 else new_image_dim[1]/blocks[1] +1

    # Get the output image
    output = numpy.zeros(new_image_dim,dtype="uint8")
    
    # Call the kernel
    copy_texture_func(
        numpy.float32(resize), numpy.float32(angle),
        numpy.uint16(a.shape[1]), numpy.uint16(a.shape[0]),
        numpy.uint16(new_image_dim[1]), numpy.uint16(new_image_dim[0]),
            cuda.Out(output),texrefs=[texref],block=blocks,grid=(gridx,gridy))
    
    return output

if __name__ == '__main__':
    import Image
    import sys
    
    def main( ):
        if len(sys.argv) != 2:
            print("You should really read the source...\n\nUsage: rotate.py <Imagename>\n")
            sys.exit(-1)

        # Open, convert to grayscale, convert to numpy array
        img = Image.open(sys.argv[1]).convert("L")
        i = numpy.fromstring(img.tostring(),dtype="uint8").reshape(img.size[1],img.size[0])
        
        # Rotate & convert back to PIL Image
        irot = rotate_image(i)
        rotimg = Image.fromarray(irot,mode="L")

        # Save and display
        rotimg.save("rotated.png")
        rotimg.show()
    
    main()



