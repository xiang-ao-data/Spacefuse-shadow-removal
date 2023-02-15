""""
To detect the shadow in spacecraft image , note that the background should be removed
"""

import PIL.Image as image
import numpy as np

def shadow_detection(img1,img2):
    img1 = img1.convert('L')
    img2 = img2.convert('L')
    img1 = np.array(img1,dtype=float)
    img_1 = np.zeros(img1.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i][j] <50:
                img_1[i][j]=1
    img2 = np.array(img2,dtype=float)
    img_2 = np.zeros(img2.shape)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if img2[i][j] <50:
                img_2[i][j]=1
                if img_1[i][j] == img_2[i][j]:
                    img_2[i][j] = img_1[i][j] = 0
    img_1 = img_1.astype(np.uint8)
    img_2 = img_2.astype(np.uint8) # astype is important, only uint8 can be transform to tensor

    return img_1, img_2 # (0,1) type mask, img_1 is the shadow mask in img1.





def test():
    a_path = './all/30/A.png'
    b_path = './all/30/B.png'
    img1 = image.open(a_path)
    img2 = image.open(b_path)
    img1 = img1.resize((256,256))
    img2 = img2.resize((256, 256))
    img_1,img_2 = shadow_detection(img1,img2) # float
    img_2 = image.fromarray(img_2)
    img_2.show()
    img_2.save("./val/8_s_1.png")


if __name__ =="__main__":
    test()