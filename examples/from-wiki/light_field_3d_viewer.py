#!python 
"""
3D display of Light Field images.
Example images can be download from:
http://graphics.stanford.edu/software/LFDisplay/LFDisplay-samples.zip

Use:
>> python LFview.py <path to light-field image>

Prerequisites:
- Python 2.7
- Enthought Tool Suite (ETS)
- PIL
- Jinja

Author: Amit Aides. amitibo at technion . ac . il
"""


from enthought.traits.api import HasTraits, Range, on_trait_change
from enthought.traits.ui.api import View, Item
from enthought.chaco.api import Plot, ArrayPlotData, gray
from enthought.enable.component_editor import ComponentEditor

import numpy as np
import Image
import argparse
import os.path
import math

import pycuda.driver as cuda
import pycuda.compiler
import pycuda.autoinit

from jinja2 import Template


_kernel_tpl = Template("""
{% if NCHANNELS == 3 %}
texture<float4, 2> tex;
{% else %}
texture<float, 2> tex;
{% endif %}

__global__ void LFview_kernel(
    float x_angle,
    float y_angle,
    unsigned char* data
    )
{
    //
    // calculate pixel idx
    //
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    //
    // We might be outside the reachable pixels. Don't do anything
    //
    if( (x >= {{newiw}}) || (y >= {{newih}}) )
        return;

    //
    // calculate offset into destination array
    //
    unsigned int didx = (y * {{newiw}} + x) * {{NCHANNELS}};
    
    //
    // calculate offset into source array (be aware of rotation and scaling)
    //
    float sx = {{x_start}} + tan(x_angle)*{{x_ratio}} + {{x_step}}*x;
    float sy = {{y_start}} + tan(y_angle)*{{y_ratio}} + {{y_step}}*y;

    if( (sx < 0) || (sx >= {{oldiw}}) || (sy < 0) || (sy >= {{oldih}}) ) {

        {% for channel in range(NCHANNELS) %}
        data[didx+{{channel}}] = 0;
        {% endfor %}

        return;
    }

    {% if NCHANNELS == 3 %}
    float4 texval = tex2D(tex, sx, sy);
    data[didx] = texval.x;
    data[didx+1] = texval.y;
    data[didx+2] = texval.z;
    {% else %}
    data[didx] = tex2D(tex, sx, sy);
    {% endif %}
}
""")


def ceil(x):
    return int(x + 0.5)


class LFapplication(HasTraits):

    traits_view = View(
        Item('LF_img', editor=ComponentEditor(), show_label=False),
        Item('X_angle', label='Angle in the X axis'),
        Item('Y_angle', label='Angle in the Y axis'),
        resizable = True,
        title="LF Image"
        )

    def __init__(self, img_path):
        super().__init__()

        #
        # Load image data
        #
        base_path = os.path.splitext(img_path)[0]
        lenslet_path = base_path + '-lenslet.txt'
        optics_path = base_path + '-optics.txt'

        with open(lenslet_path) as f:
            tmp = eval(f.readline())
            x_offset, y_offset, right_dx, right_dy, down_dx, down_dy = \
              np.array(tmp, dtype=np.float32)

        with open(optics_path) as f:
            for line in f:
                name, val = line.strip().split()
                try:
                    setattr(self, name, np.float32(val))
                except:
                    pass

        max_angle = math.atan(self.pitch/2/self.flen)

        #
        # Prepare image
        #
        im_pil = Image.open(img_path)
        if im_pil.mode == 'RGB':
            self.NCHANNELS = 3
            w, h = im_pil.size
            im = np.zeros((h, w, 4), dtype=np.float32)
            im[:, :, :3] = np.array(im_pil).astype(np.float32)
            self.LF_dim = (ceil(h/down_dy), ceil(w/right_dx), 3)
        else:
            self.NCHANNELS = 1
            im = np.array(im_pil.getdata()).reshape(im_pil.size[::-1]).astype(np.float32)
            h, w = im.shape
            self.LF_dim = (ceil(h/down_dy), ceil(w/right_dx))

        x_start = x_offset - int(x_offset / right_dx) * right_dx
        y_start = y_offset - int(y_offset / down_dy) * down_dy
        x_ratio = self.flen * right_dx / self.pitch
        y_ratio = self.flen * down_dy / self.pitch

        #
        # Generate the cuda kernel
        #
        mod_LFview = pycuda.compiler.SourceModule(
            _kernel_tpl.render(
                newiw=self.LF_dim[1],
                newih=self.LF_dim[0],
                oldiw=w,
                oldih=h,
                x_start=x_start,
                y_start=y_start,
                x_ratio=x_ratio,
                y_ratio=y_ratio,
                x_step=right_dx,
                y_step=down_dy,
                NCHANNELS=self.NCHANNELS
                )
            )
        
        self.LFview_func = mod_LFview.get_function("LFview_kernel")
        self.texref = mod_LFview.get_texref("tex")
        
        #
        # Now generate the cuda texture
        #
        if self.NCHANNELS == 3:
            cuda.bind_array_to_texref(
                cuda.make_multichannel_2d_array(im, order="C"),
                self.texref
                )
        else:
            cuda.matrix_to_texref(im, self.texref, order="C")
            
        #
        # We could set the next if we wanted to address the image
        # in normalized coordinates ( 0 <= coordinate < 1.)
        # texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
        #
        self.texref.set_filter_mode(cuda.filter_mode.LINEAR)

        #
        # Prepare the traits
        #
        self.add_trait('X_angle', Range(-max_angle, max_angle, 0.0))
        self.add_trait('Y_angle', Range(-max_angle, max_angle, 0.0))
        
        self.plotdata = ArrayPlotData(LF_img=self.sampleLF())
        self.LF_img = Plot(self.plotdata)
        if self.NCHANNELS == 3:
            self.LF_img.img_plot("LF_img")
        else:
            self.LF_img.img_plot("LF_img", colormap=gray)

    def sampleLF(self):
        #
        # Get the output image
        #
        output = np.zeros(self.LF_dim, dtype=np.uint8)
        
        #
        # Calculate the gridsize. This is entirely given by the size of our image. 
        #
        blocks = (16, 16, 1)
        gridx = ceil(self.LF_dim[1]/blocks[1])
        gridy = ceil(self.LF_dim[0]/blocks[0])
        grid = (gridx, gridy)

        #
        # Call the kernel
        #
        self.LFview_func(
            np.float32(self.X_angle),
            np.float32(self.Y_angle),
            cuda.Out(output),
            texrefs=[self.texref],
            block=blocks,
            grid=grid
            )

        return output

    @on_trait_change('X_angle, Y_angle')        
    def updateImge(self):
        self.plotdata.set_data('LF_img', self.sampleLF())
        
        
def main(img_path):
    """Main function"""

    app = LFapplication(img_path)
    app.configure_traits()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View an LF image')
    parser.add_argument('img_path', type=str, help='Path to LF image')
    args = parser.parse_args()

    main(args.img_path)

