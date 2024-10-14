import numpy as np
import cv2

shape = (256, 192)
out_name = 'output.bmp'

if __name__=='__main__':
    filename = 'benchmark_256x192/000000.png'
    onnx_name = 'paired_div2k_9481_op18_ecbsr_x2_m8c32_prelu_256x192_20240909_163220.onnx'

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # 读取y8，图片大小和输入保持一致，否则需要resize
    norm_img = img / 255.0 # 归一化
    norm_img = norm_img.astype(np.float32) # 必须转成fp32才能推理

    net = cv2.dnn.readNetFromONNX(onnx_name)

    blob = cv2.dnn.blobFromImage(
		norm_img,
		scalefactor=1.0,
		size=shape,
		swapRB=False,
		crop=False)
    net.setInput(blob)
    out = net.forward(net.getUnconnectedOutLayersNames())[0] # 获取输出
    y8_out = out.astype(np.uint8).squeeze()
    cv2.imwrite(out_name, y8_out) # 存图