import cv2
import numpy as np
    
image = cv2.imread('dgu_night_color.png', cv2.IMREAD_COLOR)  # img2numpy
height, width, bpp = image.shape
ch=3;
in_image=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
cv2.imshow('Input Image', image)


def histogram(img):  # forward transformation
    #height, width, bpp = img.shape
    MN=width*height     # 1280 X 720

    #result = np.zeros((height, width), dtype=np.uint8)  # 0으로 초기화 된 결과 이미지, np.unit8=부호없는 정수
    gray_level= np.zeros(256, np.uint32)

    # for x in range (height):
    #     temp=list(img[x]) #리스트의 픽셀 리스트값을 받는 temp
    #     for i in range (256):
    #         gray_level1[i]=gray_level1[i]+temp.count(i)

    for x in range(width):
        for y in range(height):
            i=img[y][x]
            gray_level[i]+=1
    # print(gray_level)

    CDF= np.zeros(256, np.float16)

    CDF[0]=gray_level[0]/MN
    for i in range(1,256):            
            CDF[i]=gray_level[i]/MN+CDF[i-1]
    # print(CDF)

    output_gray_level=[CDF[i]*255 for i in range(256)]
    output_gray_level=np.round(output_gray_level,0)
    print(output_gray_level)

    for x in range(width):
        for y in range(height):
            temp=ychannel[y][x]
            ychannel[y][x]=output_gray_level[temp]

    return ychannel

def color_image_processing(cvt_img,ch):
    
    temp_image = np.zeros((height,width), dtype=np.uint8)
    input_bgr_channel= np.zeros((height,width), dtype=np.uint8)
    result_image=np.zeros((height,width,3), dtype=np.uint8)
    
    for k in range(ch):
        input_bgr_channel=image[:,:,k]
        temp_image=cvt_img*((input_bgr_channel/ychannel)**0.5)
        result_image[:,:,k]=temp_image
    
    return result_image

ychannel=np.zeros((height,width),dtype=np.uint8)
ychannel=in_image[:,:,0]

out_image = histogram(ychannel)
real_out_image=color_image_processing(out_image,ch)

cv2.imshow('Result Image', real_out_image)
cv2.imwrite('dgu_night_equalization.png', real_out_image)  # save result img
#cv2.waitKey()
