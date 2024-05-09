import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtGui import *

from PyQt5.QtCore import Qt, QRect, QPoint, QSize
from PyQt5.QtGui import QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QPushButton, QVBoxLayout, QLineEdit

from infer import AdvancedMNISTInfer


def q_image_to_numpy(qimg: QImage):
    ptr = qimg.constBits()
    ptr.setsize(qimg.byteCount())

    mat = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)  # 注意这地方通道数一定要填4，否则出错
    return mat


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.draw_max_height = 220
        self.draw_min_height = 20
        # 网络区参数
        self.read_ckpt = "ckpt/test.ckpt"
        self.symbol_mapping_path = "./ckpt/symbol_mapping.npy"
        self.output_class = 16
        self.image_compress_x = 32
        self.image_compress_y = 32
        self.infer_module = AdvancedMNISTInfer(self.read_ckpt,
                                               self.symbol_mapping_path,
                                               self.output_class,
                                               self.image_compress_x,
                                               self.image_compress_y)

        # 窗口设置
        self.setWindowTitle("AdvancedMNIST")
        self.resize(1440, 720)
        # 窗口透明度
        # self.setWindowOpacity(0.5)
        # 垂直序列
        # self.layout = QVBoxLayout(self)

        # 绘画框
        # setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)
        '''
            要想将按住鼠标后移动的轨迹保留在窗体上
            需要一个列表来保存所有移动过的点
        '''
        self.pos_xy = []
        self.static_lines = []
        self._prepare_static_lines()

        # 分析按钮
        self.confirm_button = QPushButton('分析')
        self.confirm_button.setParent(self)
        self.confirm_button.setGeometry(20, 500, 100, 40)
        self.confirm_button.setGeometry(20, 500, 100, 40)
        self.confirm_button.clicked.connect(self.analyst_hand_inputs)
        # 分析输入文本框
        self.analyse_result_box = QLineEdit()
        self.analyse_result_box.setParent(self)
        self.analyse_result_box.setGeometry(150, 500, 200, 40)
        # 清除文本框按钮
        self.clear_button = QPushButton('清除文本框')
        self.clear_button.setParent(self)
        self.clear_button.setGeometry(150, 550, 100, 40)
        self.clear_button.clicked.connect(self._clear_analyse_result)
        # 清除按钮
        self.clear_button = QPushButton('清除画板')
        self.clear_button.setParent(self)
        self.clear_button.setGeometry(20, 550, 100, 40)
        self.clear_button.clicked.connect(self._clear_paint_board)
        # 计算按钮
        self.calc_button = QPushButton('计算')
        self.calc_button.setParent(self)
        self.calc_button.setGeometry(20, 600, 100, 40)
        self.calc_button.clicked.connect(self._calc_analyse_result)
        # 计算结果
        self.calc_result_box = QLineEdit()
        self.calc_result_box.setParent(self)
        self.calc_result_box.setGeometry(150, 600, 200, 40)
        # 推理网络

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 10, Qt.SolidLine)
        painter.setPen(pen)

        '''
            首先判断pos_xy列表中是不是至少有两个点了
            然后将pos_xy中第一个点赋值给point_start
            利用中间变量pos_tmp遍历整个pos_xy列表
                point_end = pos_tmp

                判断point_end是否是断点，如果是
                    point_start赋值为断点
                    continue
                判断point_start是否是断点，如果是
                    point_start赋值为point_end
                    continue

                画point_start到point_end之间的线
                point_start = point_end
            这样，不断地将相邻两个点之间画线，就能留下鼠标移动轨迹了
        '''
        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        pen = QPen(Qt.black, 4, Qt.SolidLine)
        painter.setPen(pen)
        if len(self.static_lines) > 1:
            point_start = self.static_lines[0]
            for pos_tmp in self.static_lines:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        '''
            按住鼠标移动事件：将当前点添加到pos_xy列表中
            调用update()函数在这里相当于调用paintEvent()函数
            每次update()时，之前调用的paintEvent()留下的痕迹都会清空
        '''
        # 中间变量pos_tmp提取当前点
        position_x = event.pos().x()
        position_y = event.pos().y()
        pos_tmp = (position_x, position_y)
        if self.draw_min_height < position_y < self.draw_max_height:
            # pos_tmp添加到self.pos_xy中
            self.pos_xy.append(pos_tmp)
            self.update()
        else:
            if self.pos_xy[-1][0] != -1:
                self.pos_xy.append((-1, -1))

    def mouseReleaseEvent(self, event):
        '''
            重写鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
            然后在绘画时判断一下是不是断点就行了
            是断点的话就跳过去，不与之前的连续
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        self.update()

    def _clear_paint_board(self):
        self.pos_xy = []
        self.update()

    def _clear_analyse_result(self):
        self.analyse_result_box.setText("")

    def _prepare_static_lines(self):
        self.static_lines.append((20, 20))
        self.static_lines.append((1420, 20))
        self.static_lines.append((1420, 220))
        self.static_lines.append((20, 220))
        self.static_lines.append((-1, -1))
        for i in range(7):
            self.static_lines.append((20 + 200 * i, 20))
            self.static_lines.append((20 + 200 * i, 220))
            self.static_lines.append((-1, -1))

    def _calc_analyse_result(self):
        to_be_calc_text = self.analyse_result_box.text()
        try:
            calc_result = eval(to_be_calc_text)
            self.calc_result_box.setText(str(calc_result))
        except:
            self.calc_result_box.setText("表达式有误")

    def analyst_hand_inputs(self):
        infer_results = ""
        screen_shots = self._dots_connection_to_image()
        for screen_shot in screen_shots:
            # print(screen_shot.sum())
            if screen_shot.sum() < 9000000:
                infer_result = self._infer_from_image(screen_shot)
                infer_results = infer_results + infer_result
        self.analyse_result_box.setText(self.analyse_result_box.text() + infer_results)

    def _infer_from_image(self, image: np.ndarray) -> str:
        return self.infer_module.infer_from_raw_image(image)

    def _dots_connection_to_image(self):
        screen_shots = []
        for i in range(7):
            start_x = 22 + 200 * i
            start_y = 22
            ss_q_pixmap: QPixmap = self.grab(rectangle=QRect(QPoint(start_x, start_y), QSize(196, 196)))
            ss_numpy: np.ndarray = q_image_to_numpy(ss_q_pixmap.toImage())[..., 0:3]
            ss_numpy_gray = ss_numpy.mean(axis=2)
            # print(2)
            # print(ss_numpy_gray.shape)
            # plt.imshow(ss_numpy)
            # plt.show()
            # input()
            screen_shots.append(ss_numpy_gray)
        return screen_shots


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
