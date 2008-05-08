from _driver import *




def _add_functionality():
    def device_get_attributes(dev):
        return dict((getattr(device_attribute, att), 
            dev.get_attribute(getattr(device_attribute, att))
            )
            for att in dir(device_attribute)
            if att[0].isupper())

    Device.get_attributes = device_get_attributes




_add_functionality()




def pagelocked_zeros(shape, dtype, order="C"):
    result = pagelocked_empty(shape, dtype, order)
    result.fill(0)
    return result




class SourceModule(object):
    def __init__(self, source, options=[], keep=False):
        from tempfile import mkdtemp
        tempdir = mkdtemp()
        print tempdir

        from os.path import join
        outf = open(join(tempdir, "kernel.cu"), "w")
        outf.write(source)
        outf.close()

        from subprocess import call
        result = call(["nvcc", "--cubin"] 
                + options
                + ["kernel.cu"]
            cwd=tempdir)

        data = open(join(tempdir, "kernel.cubin"), "r").read()
        self.module = module_from_buffer(data)

        if not keep:
            from os import listdir, unlink, rmdir
            for name in listdir(tempdir):
                os.unlink(join(tempdir, name))
            os.rmdir(tempdir)

        self.get_function = self.module.get_function
        self.get_global = self.module.get_global
        self.get_texref = self.module.get_texref
