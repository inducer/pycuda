import pycuda.driver as drv

def test_memory():
    import numpy
    import numpy.linalg as la
    z = numpy.random.randn(400).astype(numpy.float32)
    z_gpu = drv.mem_alloc(z.nbytes)
    drv.memcpy_htod(int(z_gpu), z)

    new_z = numpy.empty_like(z)
    drv.memcpy_dtoh(new_z, int(z_gpu))
    assert la.norm(new_z-z) == 0




def main():
    drv.init()
    for i in range(drv.Device.count()):
        dev = drv.Device(i)
        print dev.name(), dev.compute_capability()
        print dev.get_attributes()

        ctx = dev.make_context()

        print drv.mem_get_info()
        test_memory()





if __name__ == "__main__":
    main()
