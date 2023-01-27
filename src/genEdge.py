import os
import os.path as osp
import cv2
import numpy as np

path="data/DUTS/mask"
image_files=os.listdir(path)
save_path = "data/DUTS/edge"

for i in range(len(image_files)):
    image = image_files[i]
    portion = os.path.splitext(image)
    image_name = portion[0] + '.tif'
    # image_id = int(portion[0])

    data = cv2.imread(osp.join(path,image))
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    # 寻找轮廓
    draw_img = np.zeros((512, 512), dtype=np.uint8)
    contours, hier = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = cv2.drawContours(draw_img, contours, -1, (255, 255, 255), 2)
    cv2.imwrite(osp.join(save_path,image),res)
