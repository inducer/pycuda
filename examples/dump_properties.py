import pycuda.driver as drv



drv.init()
print "%d device(s) found." % drv.Device.count()

for ordinal in range(drv.Device.count()):
    dev = drv.Device(ordinal)
    print "Device #%d: %s" % (ordinal, dev.name())
    print "  Compute Capability: %d.%d" % dev.compute_capability()
    print "  Total Memory: %s KB" % (dev.total_memory()//(1024))
    atts = [(str(att), value) 
            for att, value in dev.get_attributes().iteritems()]
    atts.sort()
    
    for att, value in atts:
        print "  %s: %s" % (att, value)

