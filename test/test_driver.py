def main():
    import pycuda.driver as drv

    drv.init()
    for i in range(drv.Device.count()):
        dev = drv.Device(i)
        print dev.name(), dev.compute_capability()
        print dev.get_attributes()

        ctx = dev.make_context()




if __name__ == "__main__":
    main()
