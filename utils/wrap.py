import zxingcpp

def zxing(image, bc_format):
    try: barre_code = zxingcpp.read_barcodes(image, bc_format)[0].text
    except: barre_code = '0'
    return barre_code