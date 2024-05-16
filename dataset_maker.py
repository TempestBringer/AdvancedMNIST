import os.path
import sys
import time

import cv2
import numpy as np

from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QRect, QPoint, QSize
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QPushButton, QVBoxLayout, QLineEdit, QLabel
from matplotlib import pyplot as plt


def q_image_to_numpy(qimg: QImage):
    ptr = qimg.constBits()
    ptr.setsize(qimg.byteCount())

    mat = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)  # 注意这地方通道数一定要填4，否则出错
    return mat


class DatasetMaker(QWidget):
    def __init__(self):
        super().__init__()
        # 绘画区域
        self.draw_max_height = 700
        self.draw_min_height = 20
        self.draw_max_width = 700
        self.draw_min_width = 20
        # 窗口设置
        self.setWindowTitle("AdvancedMNIST")
        self.resize(720, 960)
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

        # 清除按钮
        self.clear_button = QPushButton('清除画板')
        self.clear_button.setParent(self)
        self.clear_button.setGeometry(20, 750, 100, 40)
        self.clear_button.clicked.connect(self._clear_paint_board)

        # 保存按钮
        self.save_button = QPushButton('保存图像')
        self.save_button.setParent(self)
        self.save_button.setGeometry(220, 750, 100, 40)
        self.save_button.clicked.connect(self._save_image)

        # 数据集根目录
        self.save_root_dir_title = QLabel("数据集根目录")
        self.save_root_dir_title.setParent(self)
        self.save_root_dir_title.setGeometry(20, 800, 70, 40)
        self.save_root_dir_box = QLineEdit()
        self.save_root_dir_box.setParent(self)
        self.save_root_dir_box.setGeometry(120, 800, 200, 40)
        self.save_root_dir_box.setText("./datasets/my")

        # 标签
        self.label_title = QLabel("数据标签")
        self.label_title.setParent(self)
        self.label_title.setGeometry(20, 850, 70, 40)
        self.label_box = QLineEdit()
        self.label_box.setParent(self)
        self.label_box.setGeometry(120, 850, 200, 40)

    def _save_image(self):
        screen_shot = self._dots_connection_to_image()
        save_path: str = self.save_root_dir_box.text()
        if not save_path.endswith("/"):
            save_path = save_path + "/"
        save_path = save_path + self.label_box.text()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not save_path.endswith("/"):
            save_path = save_path + "/"
        save_path += str(time.time()).replace(".", "")
        save_path += ".png"
        cv2.imwrite(save_path, screen_shot)

    def _dots_connection_to_image(self):
        start_x = 20
        start_y = 20
        ss_q_pixmap: QPixmap = self.grab(rectangle=QRect(QPoint(start_x, start_y), QSize(680, 680)))
        ss_numpy: np.ndarray = q_image_to_numpy(ss_q_pixmap.toImage())[..., 0:3]
        ss_numpy_gray = ss_numpy.mean(axis=2)
        return ss_numpy_gray

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
        if self.draw_min_height < position_y < self.draw_max_height \
                and self.draw_min_width < position_x < self.draw_max_width:
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
        self.static_lines.append((10, 10))
        self.static_lines.append((710, 10))
        self.static_lines.append((710, 710))
        self.static_lines.append((10, 710))
        self.static_lines.append((10, 10))
        self.static_lines.append((-1, -1))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DatasetMaker()
    window.show()
    sys.exit(app.exec_())
