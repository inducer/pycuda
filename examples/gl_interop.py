# This is an (unfinished) attempt at a GL interop demo,
# submitted by Peter Berrington.

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
import numpy, sys, time
import pycuda.driver as cuda_driver
import pycuda.gl as cuda_gl
from pycuda.compiler import SourceModule

#yeah yeah i know theres a lot of ugly global variables, ill fix it later
#this is all munged together from the CUDA SDK postprocessGL example.

initial_size = 640,640
current_size = initial_size
animate = True
window = None              # Number of the glut window.
time_of_last_draw = 0.0
time_of_last_titleupdate = 0.0
frames_per_second = 0.0
frame_counter = 0
output_texture = None # pointer to offscreen render target
(source_pbo, dest_pbo, cuda_module, invert,
 pycuda_source_pbo, pycuda_dest_pbo) = [None]*6
heading,pitch,bank = [0.0]*3

def create_PBOs(w,h):
    global source_pbo, dest_pbo, pycuda_source_pbo, pycuda_dest_pbo
    num_texels = w*h
    data = numpy.zeros((num_texels,4),numpy.uint8)
    source_pbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, source_pbo)
    glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    pycuda_source_pbo = cuda_gl.BufferObject(long(source_pbo))
    dest_pbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, dest_pbo)
    glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    pycuda_dest_pbo = cuda_gl.BufferObject(long(dest_pbo))

def destroy_PBOs():
    global source_pbo, dest_pbo, pycuda_source_pbo, pycuda_dest_pbo
    for pbo in [source_pbo, dest_pbo]:
        glBindBuffer(GL_ARRAY_BUFFER, long(pbo))
        glDeleteBuffers(1, long(pbo));
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    source_pbo,dest_pbo,pycuda_source_pbo,pycuda_dest_pbo = [None]*4

def create_texture(w,h):
    global output_texture
    output_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, output_texture)
    # set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # buffer data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                 w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0)

def destroy_texture():
    global output_texture
    glDeleteTextures(1, output_texture);
    output_texture = None

def init_gl():
    global current_size
    Width, Height = current_size
    glClearColor(0.5, 0.5, 0.5, 1.0)
    glDisable(GL_DEPTH_TEST)
    glViewport(0, 0, Width, Height)
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, Width/float(Height), 0.1, 10.0)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glEnable(GL_LIGHT0)
    red   = ( 1.0, 0.1, 0.1, 1.0 )
    white = ( 1.0, 1.0, 1.0, 1.0 )
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,  red  )
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white)
    glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 60.0)

def resize(Width, Height):
    global current_size
    current_size = Width, Height
    glViewport(0, 0, Width, Height)        # Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0,Width,Height,0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def do_tick():
    global time_of_last_titleupdate, frame_counter, frames_per_second
    if ((time.clock () * 1000.0) - time_of_last_titleupdate >= 1000.):
        frames_per_second = frame_counter                   # Save The FPS
        frame_counter = 0  # Reset The FPS Counter
        szTitle = "Hurrrrrrr - %d FPS" % (frames_per_second )
        glutSetWindowTitle ( szTitle )
        time_of_last_titleupdate = time.clock () * 1000.0
    frame_counter += 1

def idle():
    global heading, pitch, bank
    if animate:
        heading += 0.2
        pitch   += 0.6
        bank    += 1.0

    glutPostRedisplay()

def display():
    try:
        render_scene()
        process_image()
        display_image()
        glutSwapBuffers()
    except:
        from traceback import print_exc
        print_exc()
        from os import _exit
        _exit()

def process(width, height):
    """ use pycuda to munge the pbo """
    #just doing a simple inversion for now :P
    assert invert != None #depending on this function to exist by now
    grid_dimensions   = (width//16,height//16)

    source_mapping = pycuda_source_pbo.map()
    dest_mapping   = pycuda_dest_pbo.map()

    invert.prepared_call(grid_dimensions, 
            source_mapping.device_ptr(), 
            dest_mapping.device_ptr())

    cuda_driver.Context.synchronize()

    source_mapping.unmap()
    dest_mapping.unmap()

def process_image():
    """ copy image and process using CUDA """
    global pycuda_source_pbo,source_pbo,current_size, dest_pbo
    image_width, image_height = current_size
    assert source_pbo is not None

    # tell cuda we are going to get into these buffers
    pycuda_source_pbo.unregister()

    # Note: in the cuda source the pbo is unregistered and reregistered every
    # frame.  I assume pycuda pbo's unregister automatically because
    # uncommenting the above line will cause an error as pycuda attempts to
    # unregister an already unregistered PBO.

    # activate destination buffer
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(source_pbo))

    # read data into pbo. note: use BGRA format for optimal performance

    import OpenGL.raw.GL as rawgl

    rawgl.glReadPixels(
             0,                  #start x
             0,                  #start y
             image_width,        #end   x
             image_height,       #end   y
             GL_BGRA,            #format
             GL_UNSIGNED_BYTE,   #output type
             ctypes.c_void_p(0))

    pycuda_source_pbo = cuda_gl.BufferObject(long(source_pbo))

    # run the Cuda kernel
    process(image_width, image_height)
    # blit convolved texture onto the screen
    # download texture from PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, dest_pbo)
    glBindTexture(GL_TEXTURE_2D, output_texture)

    rawgl.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    image_width, image_height,
                    GL_BGRA, GL_UNSIGNED_BYTE, 0)

def display_image():
    """ render a screen sized quad """
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    glMatrixMode( GL_MODELVIEW)
    glLoadIdentity()
    glViewport(0, 0, current_size[0], current_size[1])
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0, -1.0, 0.5)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(1.0, -1.0, 0.5)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0, 1.0, 0.5)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-1.0, 1.0, 0.5)
    glEnd()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glDisable(GL_TEXTURE_2D)
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0)


def render_scene():
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)# Clear Screen And Depth Buffer
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity ()      # Reset The Modelview Matrix
    glTranslatef(0.0, 0.0, -3.0);
    glRotatef(heading, 1.0, 0.0, 0.0)
    glRotatef(pitch  , 0.0, 1.0, 0.0)
    glRotatef(bank   , 0.0, 0.0, 1.0)
    glViewport(0, 0, 512, 512)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glutSolidTeapot(1.0)
    do_tick()#just for fps display..
    return True

# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)
def keyPressed(*args):
    global animate
    # If escape is pressed, kill everything.
    if args[0] == '\033':
        print 'detected Escape.. quitting'
        destroy_PBOs()
        destroy_texture()
        exit()
    elif args[0] == 'a':
        animate = not animate # toggle whether we are currently animating

def main():
    global window, cuda_module, cuda_gl, cuda_driver, invert
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
    glutInitWindowSize(*initial_size)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow("ZZZZZZZZZZZZZZZZZZ")
    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutReshapeFunc(resize)
    glutKeyboardFunc(keyPressed)
    glutSpecialFunc(keyPressed)
    init_gl()

    # create texture for blitting to screen
    create_texture(*initial_size)

    #setup pycuda gl interop
    import pycuda.gl.autoinit
    import pycuda.gl
    cuda_gl = pycuda.gl
    cuda_driver = pycuda.driver

    cuda_module = SourceModule("""
    __global__ void invert(unsigned char *source, unsigned char *dest)
    {
      int idx = 3*(256*(blockIdx.x + blockIdx.y * 16) + 
        threadIdx.y * blockDim.x + threadIdx.x);
      dest[idx  ] = 255 - source[idx  ];
      dest[idx+1] = 255 - source[idx+1];
      dest[idx+2] = 255 - source[idx+2];
    }
    """)
    invert = cuda_module.get_function("invert")
    invert.prepare("PP", (16, 16, 1))

    # create source and destination pixel buffer objects for processing
    create_PBOs(*initial_size)

    glutMainLoop()

# Print message to console, and kick off the main to get it rolling.
if __name__ == "__main__":
    print "Hit ESC key to quit."
    main()
