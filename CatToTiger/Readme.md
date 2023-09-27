# 基于特征的图像合成

本次作业实现了两种不同的基于特征的图像识别算法

#### ①基于特征点的图像合成

文件：Img_morphing_point.py

测试结果及样例： result_face_img_point

参考连接：https://learnopencv.com/face-morph-using-opencv-cpp-python/

优缺点：使用用dlib的脚本识别出人脸的特征点后，使用人脸的图片作合成效果就极佳，但猫和虎的特征点dlib中没有对应的脚本识别特征点，采用手动标注的方式标注猫和虎的特征点后，实现出效果就不是很理想。结果集为**人脸合成示例**。

#### ②基于特征线的图像合成

文件：Img_morphing_line.py

测试结果及样例： result_img_line

参考连接：https://www.csie.ntu.edu.tw/~b97074/vfx_html/hw1.html

备注：本项目用python实现了基于特征线的图像合成算法，所标注特征线越多，合成效果越好，但是效率会明显下降。结果集为**猫变虎**。

使用方式：

1. 运行 Img_morphing_line.py 文件
2. 点击弹出的窗口，点击空格键，分别**依次**在左右两个窗口中用鼠标绘制特征线
3. 重复步骤2，每添加一条特征线**点击一次空格键**
4. 特征线绘制完成后，点击**回车键**开始合成
5. 点击**ESC键**退出程序
