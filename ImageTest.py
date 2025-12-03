from commons import Common
import Receiver2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ai method
def image_to_pixel_bytes(image_path):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    arr = np.array(img, dtype=int)  # shape (H, W, 3)
    flat = arr.reshape(-1)  # 1D array of length W*H*3
    print(len(flat))
    return flat, (width, height)


# ai method

def bytes_to_image(bits: list[int], width: int, height: int):
    # force values to literal python ints so joining works
    bits = [0 if b == 0 else 1 for b in bits]

    # pad to full bytes
    if len(bits) % 8 != 0:
        bits += [0] * (8 - (len(bits) % 8))

    # convert every 8 bits into a byte
    byte_values = []
    for i in range(0, len(bits), 8):
        chunk = bits[i:i+8]
        byte = int("".join(str(b) for b in chunk), 2)
        byte_values.append(byte)

    arr = np.asarray(byte_values, dtype=np.uint8)

    channels = 3
    expected_len = width * height * channels

    if arr.size < expected_len:
        arr = np.concatenate([arr, np.zeros(expected_len - arr.size, dtype=np.uint8)])
    elif arr.size > expected_len:
        arr = arr[:expected_len]

    arr = arr.reshape((height, width, channels))
    return Image.fromarray(arr, "RGB")





received = np.array([])


def bitsCallback(bits):
    global received
    received = np.concatenate([received, bits])
    im = bytes_to_image(received, 10,10)
    print(bits)
    #plt.imshow(im)

    im.save("im.jpg")


path = "images/sunset"
def main():

    global received
    pix, size = image_to_pixel_bytes(path)
    #orig = bytes_to_image(pix, size[0], size[1])

    global im  
    im = pix

    #plt.imshow(orig)
    #plt.axis("off")  # hide axes

    rcv = Receiver2.Receiver(Common.config, Common.mod, Common.demod, bitsCallback)

    rcv.rcv.listen()

    #plt.show()



if __name__ == "__main__":
    main()
