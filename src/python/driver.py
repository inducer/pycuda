from _driver import *




def _add_functionality():
    def device_get_attributes(dev):
        return dict((getattr(device_attribute, att), 
            dev.get_attribute(getattr(device_attribute, att))
            )
            for att in dir(device_attribute)
            if att[0].isupper())

    Device.get_attributes = device_get_attributes




def pagelocked_zeros(shape, dtype, order="C"):
    result = pagelocked_empty(shape, dtype, order)
    result.fill(0)
    return result




_add_functionality()
