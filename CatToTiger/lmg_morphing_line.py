import cv2
import numpy as np
from PIL import Image as Imm
counter = 0
frame_count = 10
left_image = None    # 第一张图片
right_image = None   # 第二张图片
left_image_tmp = None
right_image_tmp = None
height = 0
width = 0
Color = (0, 255, 0)  # 颜色
Thickness = 2  # 线宽
Shift = 0  # 位移
key = 0

parameter_a = 1.0
parameter_b = 2.0
parameter_p = 0.0

first_image_name = "./result_img_line/cat.png"   # 第一张图片名称
second_image_name = "./result_img_line/tiger.png"  # 第二张图片名称
new_image_name = "./result_img_line/result"
image_files =[]  # 读取图像文件名

pairs = []          # 线段队的集合
curLinePair = None  # 正在处理的线段对

# 结构体 Line
class Line:
    def __init__(self):
        self.P = (0.0, 0.0)  # start
        self.Q = (0.0, 0.0)  # end
        self.M = (0.0, 0.0)  # mid
        self.len = 0.0
        self.degree = 0.0

    def PQtoMLD(self):  # 线段首尾到中点及斜率
        self.M = ((self.P[0] + self.Q[0]) / 2, (self.P[1] + self.Q[1]) / 2)
        tmpx = self.Q[0] - self.P[0]
        tmpy = self.Q[1] - self.P[1]
        self.len = np.sqrt(tmpx * tmpx + tmpy * tmpy)
        self.degree = np.arctan2(tmpy, tmpx)

    def MLDtoPQ(self): # 线段中点斜率到线段首尾
        tmpx = 0.5 * self.len * np.cos(self.degree)
        tmpy = 0.5 * self.len * np.sin(self.degree)
        self.P = (self.M[0] - tmpx, self.M[1] - tmpy)
        self.Q = (self.M[0] + tmpx, self.M[1] + tmpy)

    def show(self):
        print(f"P({self.P[0]},{self.P[1]}) Q({self.Q[0]},{self.Q[1]}) M({self.M[0]},{self.M[1]})")
        print(f"len={self.len} degree={self.degree}")

    def Getu(self, X):
        X_P_x = X[0] - self.P[0]
        X_P_y = X[1] - self.P[1]
        Q_P_x = self.Q[0] - self.P[0]
        Q_P_y = self.Q[1] - self.P[1]
        u = ((X_P_x * Q_P_x) + (X_P_y * Q_P_y)) / (self.len * self.len)
        return u

    def Getv(self, X):
        X_P_x = X[0] - self.P[0]
        X_P_y = X[1] - self.P[1]
        Q_P_x = self.Q[0] - self.P[0]
        Q_P_y = self.Q[1] - self.P[1]
        Perp_Q_P_x = Q_P_y
        Perp_Q_P_y = -Q_P_x
        v = ((X_P_x * Perp_Q_P_x) + (X_P_y * Perp_Q_P_y)) / self.len
        return v

    def Get_Point(self, u, v):
        Q_P_x = self.Q[0] - self.P[0]
        Q_P_y = self.Q[1] - self.P[1]
        Perp_Q_P_x = Q_P_y
        Perp_Q_P_y = -Q_P_x
        Point_x = self.P[0] + u * Q_P_x + ((v * Perp_Q_P_x) / self.len)
        Point_y = self.P[1] + u * Q_P_y + ((v * Perp_Q_P_y) / self.len)
        return (Point_x, Point_y)

    def Get_Weight(self, X):
        a = parameter_a
        b = parameter_b
        p = parameter_p
        d = 0.0

        u = self.Getu(X)
        if u > 1.0:
            d = np.sqrt((X[0] - self.Q[0]) * (X[0] - self.Q[0]) + (X[1] - self.Q[1]) * (X[1] - self.Q[1]))
        elif u < 0:
            d = np.sqrt((X[0] - self.P[0]) * (X[0] - self.P[0]) + (X[1] - self.P[1]) * (X[1] - self.P[1]))
        else:
            d = abs(self.Getv(X))

        weight = np.power(np.power(self.len, p) / (a + d), b)
        return weight

# 结构体 LinePair
class LinePair:
    def __init__(self):
        self.leftLine = Line()
        self.rightLine = Line()
        self.warpLine = []

    def genWarpLine(self):
        while self.leftLine.degree - self.rightLine.degree > 3.14159265:
            self.rightLine.degree = self.rightLine.degree + 3.14159265
        while self.rightLine.degree - self.leftLine.degree > 3.14159265:
            self.leftLine.degree = self.leftLine.degree + 3.14159265
        for i in range(frame_count):  # 中点，长度，斜率合成，按比例
            ratio = (i + 1) / (frame_count + 1.0)
            curLine = Line()

            curLine.M = ((1 - ratio) * self.leftLine.M[0] + ratio * self.rightLine.M[0],
                         (1 - ratio) * self.leftLine.M[1] + ratio * self.rightLine.M[1])
            curLine.len = (1 - ratio) * self.leftLine.len + ratio * self.rightLine.len
            curLine.degree = (1 - ratio) * self.leftLine.degree + ratio * self.rightLine.degree

            curLine.MLDtoPQ()
            self.warpLine.append(curLine)

class Image:
    def __init__(self, i):
        self.frame_index = i
        self.new_image = cv2.imread(first_image_name)
        self.test_image = cv2.imread(first_image_name)

    def bilinear(self, image, X, Y):
        x_floor = int(X)
        y_floor = int(Y)
        x_ceil = x_floor + 1
        y_ceil = y_floor + 1
        a = X - x_floor
        b = Y - y_floor

        if x_ceil >= width - 1:
            x_ceil = width - 1
        if y_ceil >= height - 1:
            y_ceil = height - 1

        output_scalar = np.zeros(4, dtype=np.uint8)
        leftdown = image[y_floor, x_floor]
        lefttop = image[y_ceil, x_floor]
        rightdown = image[y_floor, x_ceil]
        righttop = image[y_ceil, x_ceil]

        for i in range(3):
            output_scalar[i] = int(
                (1 - a) * (1 - b) * leftdown[i] +
                a * (1 - b) * rightdown[i] +
                a * b * righttop[i] +
                (1 - a) * b * lefttop[i]
            )

        return output_scalar

    def Warp(self):
        ratio = (self.frame_index + 1) / (frame_count + 1)
        ori_left_image = cv2.imread(first_image_name)
        ori_right_image = cv2.imread(second_image_name)

        for x in range(width):
            for y in range(height):
                dst_point = (x, y)
                left_x_sum_x = 0.0
                left_x_sum_y = 0.0
                left_weight_sum = 0.0
                right_x_sum_x = 0.0
                right_x_sum_y = 0.0
                right_weight_sum = 0.0

                for i in range(len(pairs)):
                    src_line = pairs[i].leftLine
                    dst_line = pairs[i].warpLine[self.frame_index]

                    new_u = dst_line.Getu(dst_point)
                    new_v = dst_line.Getv(dst_point)

                    src_point = src_line.Get_Point(new_u, new_v)
                    src_weight = dst_line.Get_Weight(dst_point)
                    left_x_sum_x += src_point[0] * src_weight
                    left_x_sum_y += src_point[1] * src_weight
                    left_weight_sum += src_weight

                    src_line = pairs[i].rightLine

                    new_u = dst_line.Getu(dst_point)
                    new_v = dst_line.Getv(dst_point)

                    src_point = src_line.Get_Point(new_u, new_v)
                    src_weight = dst_line.Get_Weight(dst_point)
                    right_x_sum_x += src_point[0] * src_weight
                    right_x_sum_y += src_point[1] * src_weight
                    right_weight_sum += src_weight

                left_src_x = left_x_sum_x / left_weight_sum
                left_src_y = left_x_sum_y / left_weight_sum
                right_src_x = right_x_sum_x / right_weight_sum
                right_src_y = right_x_sum_y / right_weight_sum

                if left_src_x < 0:
                    left_src_x = 0
                if left_src_y < 0:
                    left_src_y = 0
                if left_src_x >= width:
                    left_src_x = width - 1
                if left_src_y >= height:
                    left_src_y = height - 1
                if right_src_x < 0:
                    right_src_x = 0
                if right_src_y < 0:
                    right_src_y = 0
                if right_src_x >= width:
                    right_src_x = width - 1
                if right_src_y >= height:
                    right_src_y = height - 1

                left_scalar = self.bilinear(ori_left_image, left_src_x, left_src_y)
                right_scalar = self.bilinear(ori_right_image, right_src_x, right_src_y)
                #new_scalar = ((1 - ratio) * left_scalar + ratio * right_scalar).astype(np.uint8)
                new_scalar = (1 - ratio) * left_scalar[0] + ratio * right_scalar[0], (1 - ratio) * left_scalar[
                    1] + ratio * right_scalar[1], (1 - ratio) * left_scalar[2] + ratio * right_scalar[2]
                new_left_scalar=left_scalar[0],left_scalar[1],left_scalar[2]
                self.new_image[y, x] = new_scalar
                self.test_image[y, x] = new_left_scalar

        win_name = f"frame[{self.frame_index}]"
        img_name = f"{new_image_name}_{self.frame_index}.jpg"
        image_files.append(img_name)
        cv2.imwrite(img_name, self.new_image)




def runWarp():
    for i in range(frame_count):
        curImage = Image(i)
        print(f"warping {i}...")
        curImage.Warp()
    # 设置GIF文件的名称和帧速率
    output_gif = "./result_img_line/result_output.gif"
    frame_rate = 5  # 帧速率，每秒10帧
    # 读取图像并将它们添加到一个列表中
    images = [Imm.open(image_file) for image_file in image_files]
    # 将图像列表保存为GIF文件
    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=1000 / frame_rate, loop=0)
    print(f"GIF已保存为 {output_gif}")


# 打印坐标
def show_pairs():
    for i in range(len(pairs)):
        print("leftLine:")
        pairs[i].leftLine.show()
        print("rightLine:")
        pairs[i].rightLine.show()
        print("\n")

def on_mousel(event, x, y, flags, param):
    global counter, left_image, left_image_tmp, pairs, curLinePair
    if counter % 2 == 0 and counter > 0:
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:  # 鼠标点击
            print(f"Left image( {x}, {y}) Event: {event} Flags: {flags} Param: {param}")
            curLinePair = LinePair()
            pairs.append(curLinePair)
            pairs[-1].leftLine.P = (x, y)
        if event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:      # 鼠标松开
            print(f"Left image( {x}, {y}) Event: {event} Flags: {flags} Param: {param}")
            pairs[-1].leftLine.Q = (x, y)
            pairs[-1].leftLine.PQtoMLD()   # 结束调用终止符号
            cv2.line(left_image, (int(pairs[-1].leftLine.P[0]), int(pairs[-1].leftLine.P[1])),
                     (int(pairs[-1].leftLine.Q[0]), int(pairs[-1].leftLine.Q[1])), Color, Thickness, cv2.LINE_AA,
                     Shift)
            left_image_tmp = left_image.copy()
            counter -= 1
        if flags == cv2.EVENT_FLAG_LBUTTON or flags == cv2.EVENT_FLAG_RBUTTON:  # 鼠标移动
            pairs[-1].leftLine.Q = (x, y)
            left_image = left_image_tmp.copy()
            cv2.line(left_image, (int(pairs[-1].leftLine.P[0]), int(pairs[-1].leftLine.P[1])),
                     (int(pairs[-1].leftLine.Q[0]), int(pairs[-1].leftLine.Q[1])), Color, Thickness, cv2.LINE_AA, Shift)
            cv2.imshow("left", left_image)


def on_mouser(event, x, y, flags, param):
    global counter, right_image, right_image_tmp, pairs, curLinePair
    if counter % 2 == 1 and counter > 0:
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            print(f"Right image( {x}, {y}) Event: {event} Flags: {flags} Param: {param}")
            curLinePair.rightLine.P = (x, y)
        if event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            print(f"Right image( {x}, {y}) Event: {event} Flags: {flags} Param: {param}")
            curLinePair.rightLine.Q = (x, y)
            curLinePair.rightLine.PQtoMLD()   # 结束调用取中点
            cv2.line(right_image, (int(curLinePair.rightLine.P[0]), int(curLinePair.rightLine.P[1])),
                     (int(curLinePair.rightLine.Q[0]), int(curLinePair.rightLine.Q[1])), Color, Thickness,
                     cv2.LINE_AA, Shift)
            right_image_tmp = right_image.copy()
            counter -= 1
            curLinePair.genWarpLine()
            pairs.append(curLinePair)
            print("\n")
            show_pairs()
        if flags == cv2.EVENT_FLAG_LBUTTON or flags == cv2.EVENT_FLAG_RBUTTON:
            curLinePair.rightLine.Q = (x, y)
            right_image = right_image_tmp.copy()
            cv2.line(right_image, (int(curLinePair.rightLine.P[0]), int(curLinePair.rightLine.P[1])),
                     (int(curLinePair.rightLine.Q[0]), int(curLinePair.rightLine.Q[1])), Color, Thickness,
                     cv2.LINE_AA, Shift)
            cv2.imshow("right", right_image)


if __name__ == "__main__":
    left_image = cv2.imread(first_image_name)
    right_image = cv2.imread(second_image_name)
    # 将右图转为左图大小
    right_image = cv2.resize(right_image,(left_image.shape[1],left_image.shape[0]),interpolation=cv2.INTER_CUBIC)

    # 读取图片的长和宽，坐标轴y轴为平面直角坐标系反向
    height, width, _ = left_image.shape
    left_image_tmp = left_image.copy()
    right_image_tmp = right_image.copy()

    cv2.namedWindow("left", cv2.WINDOW_NORMAL)
    cv2.namedWindow("right", cv2.WINDOW_NORMAL)
    cv2.moveWindow("left", 10, 10)
    cv2.moveWindow("right", 300, 10)
    cv2.resizeWindow("left", width, height)  # 设置左窗口大小为图片的大小
    cv2.resizeWindow("right", width, height)  # 设置右窗口大小为图片的大小


    cv2.setMouseCallback("left", on_mousel)  # 左侧图像的鼠标事件
    cv2.setMouseCallback("right", on_mouser)  # 右侧图像的鼠标事件

    cv2.imshow("left", left_image)
    cv2.imshow("right", right_image)

    while True:
        key = cv2.waitKey(0)
        if key == 32:  # 空格键增加新特征线
            counter += 2
        elif key == 13:  # 回车键开始合成
            runWarp()
        elif key == 27:  # esc键
            break

    cv2.destroyAllWindows()