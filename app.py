import sys
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import torch
import time
from collections import deque, defaultdict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox, 
                             QCheckBox, QRubberBand, QDesktopWidget, QMenu, QAction, 
                             QSlider, QDialog, QProgressDialog, QSizeGrip, QSpinBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex, QRect, QPoint, QSize, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap, QScreen, QPainter, QPen, QColor, QFont
from PIL import Image
import urllib.request
import importlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException
import colorsys
import threading
import logging
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
import pandas as pd
import torch.nn.functional as F

# 将当前目录添加到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 现在可以导入src模块了
from src.tracker import ObjectTracker
from src.window_capture_manager import WindowCaptureThread

# 设置日志记录
logging.basicConfig(filename='app.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferenceThread(QThread):
    inference_finished = pyqtSignal(np.ndarray, object)

    def __init__(self, model, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        self.running = False
        self.mutex = QMutex()
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_det = 300
        self.frame_queue = deque(maxlen=1)

    def run(self):
        logger.info("Inference thread started")
        self.running = True
        while self.running:
            with QMutexLocker(self.mutex):
                if self.frame_queue:
                    frame = self.frame_queue.popleft()
                else:
                    continue
            try:
                results = self.model(frame, device=self.device, conf=self.confidence_threshold, 
                                     iou=self.iou_threshold, max_det=self.max_det)
                self.inference_finished.emit(frame, results)
            except Exception as e:
                logger.error(f"Error during inference: {str(e)}")
        logger.info("Inference thread stopped")

    def update_frame(self, frame):
        self.mutex.lock()
        self.frame_queue.append(frame)
        self.mutex.unlock()

    def stop(self):
        self.running = False

    def update_parameters(self, conf, iou, max_det):
        self.confidence_threshold = conf
        self.iou_threshold = iou
        self.max_det = max_det
        if hasattr(self.model, 'conf'):
            self.model.conf = conf
        if hasattr(self.model, 'iou'):
            self.model.iou = iou
        if hasattr(self.model, 'max_det'):
            self.model.max_det = max_det
        logger.info(f"Updated inference thread parameters: conf={conf}, iou={iou}, max_det={max_det}")

class ScreenCaptureWidget(QWidget):
    capture_finished = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.rubberband = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()
        self.current = QPoint()
        self.screen = QApplication.primaryScreen().grabWindow(QDesktopWidget().winId())
        self.initial_rect = None

    def set_initial_rect(self, rect):
        self.initial_rect = rect
        self.origin = rect.topLeft()
        self.current = rect.bottomRight()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.screen)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        if self.initial_rect:
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(self.initial_rect)
        elif not self.origin.isNull() and not self.current.isNull():
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(QRect(self.origin, self.current))

    def mousePressEvent(self, event):
        self.origin = event.pos()
        self.current = self.origin
        self.rubberband.setGeometry(QRect(self.origin, self.current))
        self.rubberband.show()
        self.initial_rect = None
        self.update()

    def mouseMoveEvent(self, event):
        self.current = event.pos()
        self.rubberband.setGeometry(QRect(self.origin, self.current).normalized())
        self.update()

    def mouseReleaseEvent(self, event):
        self.rubberband.hide()
        current = event.pos()
        capture_rect = QRect(self.origin, current).normalized()
        if capture_rect.width() > 0 and capture_rect.height() > 0:
            self.capture_finished.emit(capture_rect)
        self.close()

# 添加一个新的类来创建可调整大小的透明窗口
class ResizableTransparentWindow(QWidget):
    capture_updated = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(QApplication.primaryScreen().geometry())
        
        self.rubberband = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()
        self.current = QPoint()
        self.dragging = False
        self.resizing = False
        self.initial_selection = True

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.initial_selection:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        else:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.SolidLine))
            painter.drawRect(self.rect())

    def mousePressEvent(self, event):
        if self.initial_selection:
            self.origin = event.pos()
            self.current = self.origin
            self.rubberband.setGeometry(QRect(self.origin, self.current))
            self.rubberband.show()
        else:
            self.dragging = True
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        if self.initial_selection:
            self.current = event.pos()
            self.rubberband.setGeometry(QRect(self.origin, self.current).normalized())
        elif self.dragging:
            delta = event.pos() - self.drag_start_position
            self.move(self.pos() + delta)
            self.capture_updated.emit(self.geometry())

    def mouseReleaseEvent(self, event):
        if self.initial_selection:
            self.rubberband.hide()
            selected_rect = QRect(self.origin, self.current).normalized()
            if selected_rect.width() > 10 and selected_rect.height() > 10:
                self.setGeometry(selected_rect)
                self.initial_selection = False
                self.show()
                self.capture_updated.emit(self.geometry())
            else:
                self.close()  
        else:
            self.dragging = False
            self.capture_updated.emit(self.geometry())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.initial_selection:
            self.capture_updated.emit(self.geometry())

class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 添加版本号属性
        self.version = "2.1.0"  # 当前版本号
        
        self.setWindowTitle(f"YOLOV11对象实时检测应用@文抑青年 v{self.version}")
        
        # 设置默认窗口大小和最小窗口大小
        self.default_size = QSize(1000, 700)
        self.setGeometry(100, 100, self.default_size.width(), self.default_size.height())
        self.setMinimumSize(800, 600)  # 设置最小窗口大小
        
        # 允许调整窗口大小
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # 保存窗口状态
        self.previous_state = self.windowState()
        self.previous_geometry = self.geometry()
        
        # 创建中央部件并设置大小策略
        self.central_widget = QWidget()
        self.central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # 设置图像标签的大小策略和最小大小
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(640, 480)  # 设置最小大小
        self.layout.addWidget(self.image_label, 7)  # 设置拉伸因子为7
        
        # 初始化控件
        self.model_combo = QComboBox()
        self.populate_model_combo()
        self.gpu_checkbox = QCheckBox("使用GPU")
        self.gpu_checkbox.setChecked(torch.cuda.is_available())
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.start_inference)
        self.start_button.setEnabled(False)
        self.stop_button = QPushButton("停")
        self.stop_button.clicked.connect(self.stop)
        self.save_result_button = QPushButton("保存结果")
        self.save_result_button.clicked.connect(self.save_result)
        self.save_result_button.setEnabled(False)
        self.tracking_checkbox = QCheckBox("启用目标追踪")
        self.tracking_checkbox.setChecked(False)  # 默认不启用
        self.tracking_checkbox.setEnabled(False)  # 默认禁用
        self.toggle_stats_button = QPushButton("隐藏统计数据")
        self.toggle_stats_button.clicked.connect(self.toggle_statistics_display)
        self.export_stats_button = QPushButton("导出统计数据")
        self.export_stats_button.clicked.connect(self.export_statistics)
        self.export_stats_button.setEnabled(False)

        # 使用QGridLayout替代QHBoxLayout
        self.button_layout = QGridLayout()
        self.button_layout.setSpacing(5)  # 减小按钮之间的间距
        self.button_widget = QWidget()
        self.button_widget.setLayout(self.button_layout)
        self.button_widget.setFixedHeight(150)  # 设置按钮区域的固定高度
        self.layout.addWidget(self.button_widget, 1)  # 设置拉伸因子为1

        # 在初始化控件部分添加新的复选框
        self.show_boxes_checkbox = QCheckBox("显示边界框")
        self.show_boxes_checkbox.setChecked(True)  # 默认显示边界框
        
        # 创建按钮并添加到网格布局中
        buttons = [
            (self.model_combo, 0, 0),
            (QPushButton("选择本地模型", clicked=self.load_local_model), 0, 1),
            (self.gpu_checkbox, 0, 2),
            (QPushButton("启动摄像头", clicked=self.start_camera), 0, 3),
            (QPushButton("加载图片/视频", clicked=self.load_video), 1, 0),
            (self.start_button, 1, 1),
            (self.stop_button, 1, 2),
            (QPushButton("捕捉/调整屏幕", clicked=self.adjust_screen_capture), 1, 3),
            (self.save_result_button, 2, 0),
            (self.tracking_checkbox, 2, 1),
            (self.show_boxes_checkbox, 2, 2),  # 移动到这里，紧跟在tracking_checkbox后面
            (QPushButton("设置计数线", clicked=self.set_counting_line), 2, 3),
            (QPushButton("清除计数线", clicked=self.clear_counting_line), 2, 4),  # 新增的按钮
            (QPushButton("重置统计数据", clicked=self.reset_statistics), 3, 0),
            (self.toggle_stats_button, 3, 1),
            (self.export_stats_button, 3, 2),  # 将export_stats_button移到这里
            (QPushButton("捕获进程窗口", clicked=self.start_window_capture), 0, 4),
        ]

        for widget, row, col in buttons:
            if isinstance(widget, str):
                widget = QPushButton(widget)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 改变大小策略
            widget.setFixedHeight(30)  # 设置按钮的固定高度
            self.button_layout.addWidget(widget, row, col)

        # 设置行和列的拉伸因子
        for i in range(self.button_layout.rowCount()):
            self.button_layout.setRowStretch(i, 1)
        for i in range(self.button_layout.columnCount()):
            self.button_layout.setColumnStretch(i, 1)

        self.model = None
        self.device = None
        self.base_dir = 'detection_results'
        os.makedirs(self.base_dir, exist_ok=True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.inference_thread = None
        self.frame_buffer = deque(maxlen=1)
        self.fps = 30  # 帧率
        self.frame_time = 1000 / self.fps  # 每帧的时间（毫秒）
        self.last_frame_time = 0
        self.last_annotated_frame = None

        self.screen_capture_widget = None
        self.captured_pixmap = None
        self.screen_capture_timer = QTimer(self)
        self.screen_capture_timer.timeout.connect(self.update_screen_capture)
        self.capture_rect = None

        self.last_frame = None
        self.frame_ready = False
        self.display_timer = QTimer(self)
        self.display_timer.timeout.connect(self.update_display)
        self.display_timer.start(33)  # 约30 FPS

       
        print(f"Screen DPI: {QApplication.primaryScreen().devicePixelRatio()}")
        print(f"Screen geometry: {QDesktopWidget().screenGeometry(QDesktopWidget().primaryScreen())}")

        self.class_names = None  # 用于存储类别名称

        self.menu_bar = self.menuBar()
        self.create_model_control_menu()
        self.create_about_menu()  # 添加这行来创建关于菜单
        
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_det = 300
        self.resolution = 640  # 添加分辨率设置
        self.use_tta = False   # 添加TTA设置
        
        self.result_buffer = deque(maxlen=5)  # 增加缓冲区大小

        self.max_boxes = 30  # 最显示的边界框数量
        self.overlap_threshold = 0.5  # 叠阈值

        try:
            ultralytics = importlib.import_module('ultralytics')
            YOLO = ultralytics.YOLO
            print(f"ultralytics 版本: {ultralytics.__version__}")
        except ImportError as e:
            print(f"无法导入 ultralytics: {str(e)}")
            QMessageBox.critical(self, "错误", f"无导 ultralytics 包: {str(e)}\n请确保已正确安装 ultralytics。")
            self.model_combo.setEnabled(False)
            self.load_model_button.setEnabled(False)

        # 在 ObjectDetectionApp 类中添加一个新的属性来跟踪当前模型类型
        self.current_model_type = None
        self.class_colors = {}  # 用于存储每个类别的固定颜色

        self.inference_active = False  # 添加这行
        
        # 将这行：
        # self.display_timer.start(33)  # 约30 FPS
        # 改为：
        self.display_timer = QTimer(self)
        self.display_timer.timeout.connect(self.update_display)

        self.capture_window = None
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.update_screen_capture)

        self.is_capturing = False
        self.is_inferencing = False

        self.preview_image = None
        self.is_image = False

        self.current_video_path = None

        # 设置 FFmpeg 路径
        ffmpeg_path = os.path.join(os.getcwd(), 'FFmpeg', 'ffmpeg.exe')
        if os.path.exists(ffmpeg_path):
            os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
        else:
            logger.warning(f"FFmpeg not found at {ffmpeg_path}. Using system FFmpeg if available.")

        self.counting_line = None
        self.object_tracks = defaultdict(list)
        self.count_up = 0
        self.count_down = 0
        self.track_id = 0
        
        self.temp_pixmap = None  # 存储临时图像

        self.rubber_band = None
        self.origin = QPoint()

        self.class_counts = {}  # 存储每个类别的计数
        self.class_line_counts = {}  # 存储每个类别经过计数线的总次数
        self.max_class_counts = {}  # 存储每个类别的最大计数
        self.detection_results = []  # 用于存储检测结果

        # 设置模型文件夹路径
        self.models_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        self.is_setting_line = False
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.mousePressEvent
        self.image_label.mouseMoveEvent = self.mouseMoveEvent
        self.image_label.mouseReleaseEvent = self.mouseReleaseEvent

        # 初始化目标追踪参数
        self.max_trajectory_length = 30
        self.max_age = 10
        self.min_hits = 3
        self.iou_threshold = 0.3

        # 初始化 ObjectTracker
        self.object_tracker = ObjectTracker(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)

        # 创建设置菜单
        self.create_settings_menu()

        # 添加这行来初始化 show_statistics 属性
        self.show_statistics = True

        self.window_capture_thread = None

        self.image_scale = 1.0
        self.rubber_band = None
        self.is_setting_line = False
        self.counting_line = None
        self.show_statistics = True
        self.class_counts = {}
        self.max_class_counts = {}
        self.class_line_counts = {}
        self.count_up = 0
        self.count_down = 0

    def changeEvent(self, event):
        """处理窗口状态改变事件"""
        if event.type() == event.WindowStateChange:
            if self.windowState() & Qt.WindowMaximized:
                # 窗口最大化时保存之前的状态
                if not (self.previous_state & Qt.WindowMaximized):
                    self.previous_geometry = self.normalGeometry()
            elif self.previous_state & Qt.WindowMaximized:
                # 从最大化恢复时还原到之前的大小
                self.setGeometry(self.previous_geometry)
            
            self.previous_state = self.windowState()
        super().changeEvent(event)

    def resizeEvent(self, event):
        """处理窗口大小改变事件"""
        super().resizeEvent(event)
        # 保持图像显示区域的纵横比
        if self.image_label.pixmap():
            self.display_frame(self.last_annotated_frame if self.last_annotated_frame is not None 
                             else self.frame_buffer[-1] if self.frame_buffer else None)

    def create_model_control_menu(self):
        model_menu = self.menu_bar.addMenu("模型参数设置")
        
        adjust_params_action = QAction("检测参数调整", self)
        adjust_params_action.triggered.connect(self.show_param_dialog)
        model_menu.addAction(adjust_params_action)

        adjust_display_action = QAction("调整显示", self)
        adjust_display_action.triggered.connect(self.show_display_dialog)
        model_menu.addAction(adjust_display_action)

    def create_about_menu(self):
        about_menu = self.menu_bar.addMenu("关于")
        
        version_action = QAction("版本信息", self)
        version_action.triggered.connect(self.show_version_info)
        about_menu.addAction(version_action)

    def create_settings_menu(self):
        settings_menu = self.menu_bar.addMenu("目标追踪设置")
        
        tracking_settings_action = QAction("目标追踪设置", self)
        tracking_settings_action.triggered.connect(self.show_tracking_settings)
        settings_menu.addAction(tracking_settings_action)

    def show_version_info(self):
        version_info = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <h2 style="color: #4a4a4a;">YOLOV11对象实时检测应用</h2>
            <p><b>版本:</b> {self.version}</p>
            <p><b>作者:</b> 文抑青年</p>
            <br>
            <p>本应用基于YOLOV11模型，用于实时对象检测、分割和分类。<br>
            支持屏幕捕捉画面输入，摄像头输入、图片处理和视频分析。<br>
            支持统计监测数据以及导出。</p>
            <br>
            <h3 style="color: #4a4a4a;">更新日志</h3>
            <p><b>2024/10/5 v2.0.0</b></p>
            <ul>
                <li>首次发布</li>
                <li>支持YOLOv11模型加载和实时检测</li>
                <li>支持摄像头和屏幕捕捉</li>
            </ul>
            <p><b>2024/10/28 v2.1.0 Beta</b></p>
            <ul>
                <li>优化UI界面</li>
                <li>添加目标追踪轨迹功能</li>
                <li>修复已知bug</li>
                <li>添加进程窗口捕捉功能</li>
            </ul>
            <br>
            <p><b>官方GitHub:</b> <a href='https://github.com/ultralytics/ultralytics'>https://github.com/ultralytics/ultralytics</a></p>
            <p><b>作者B站主页:</b> <a href='https://space.bilibili.com/259012968'>https://space.bilibili.com/259012968</a></p>
            <p><b>AI技术交流群:</b> <a href='https://qm.qq.com/q/aaYSwOFExG'>785400897</a></p>
            <br>
            <h3 style="color: #4a4a4a;">赞助支持</h3>
            <p>如果觉得这个项目对您有帮助，欢迎赞助支持作者继续开发维护~</p>
            <img src='zanzhu/weixin.png' width='200'><img src='zanzhu/zhifubao.jpg' width='200'>
            <br>
            <p style="font-size: 0.9em; color: #888;">© 2024/10/5 All Rights Reserved</p>
        </body>
        </html>
        """
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("关于")
        msg_box.setText(version_info)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setTextInteractionFlags(Qt.TextBrowserInteraction)
        
        # 设置弹窗大小
        msg_box.setStyleSheet("QLabel{min-width: 500px;}")
        
        
        msg_box.exec_()

    def show_param_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("调整模型参数")
        layout = QVBoxLayout()

        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_label = QLabel("置信度阈值:")
        conf_slider = QSlider(Qt.Horizontal)
        conf_slider.setRange(0, 100)
        conf_slider.setValue(int(self.confidence_threshold * 100))
        conf_value = QLabel(f"{self.confidence_threshold:.2f}")
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(conf_slider)
        conf_layout.addWidget(conf_value)
        layout.addLayout(conf_layout)

        # IOU阈值
        iou_layout = QHBoxLayout()
        iou_label = QLabel("IOU阈值:")
        iou_slider = QSlider(Qt.Horizontal)
        iou_slider.setRange(0, 100)
        iou_slider.setValue(int(self.iou_threshold * 100))
        iou_value = QLabel(f"{self.iou_threshold:.2f}")
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(iou_slider)
        iou_layout.addWidget(iou_value)
        layout.addLayout(iou_layout)

        # 最大检测数
        max_det_layout = QHBoxLayout()
        max_det_label = QLabel("最大检测数:")
        max_det_slider = QSlider(Qt.Horizontal)
        max_det_slider.setRange(1, 1000)
        max_det_slider.setValue(self.max_det)
        max_det_value = QLabel(f"{self.max_det}")
        max_det_layout.addWidget(max_det_label)
        max_det_layout.addWidget(max_det_slider)
        max_det_layout.addWidget(max_det_value)
        layout.addLayout(max_det_layout)

        # 分辨率设置
        resolution_layout = QHBoxLayout()
        resolution_label = QLabel("分辨率:")
        resolution_slider = QSlider(Qt.Horizontal)
        resolution_slider.setRange(320, 1280)
        resolution_slider.setValue(self.resolution)
        resolution_value = QLabel(f"{self.resolution}")
        resolution_layout.addWidget(resolution_label)
        resolution_layout.addWidget(resolution_slider)
        resolution_layout.addWidget(resolution_value)
        layout.addLayout(resolution_layout)

       
        tta_layout = QHBoxLayout()
        tta_label = QLabel("使用TTA:")
        tta_checkbox = QCheckBox()
        tta_checkbox.setChecked(self.use_tta)
        tta_layout.addWidget(tta_label)
        tta_layout.addWidget(tta_checkbox)
        layout.addLayout(tta_layout)

        def update_conf(value):
            self.confidence_threshold = value / 100
            conf_value.setText(f"{self.confidence_threshold:.2f}")
            self.update_inference_parameters()

        def update_iou(value):
            self.iou_threshold = value / 100
            iou_value.setText(f"{self.iou_threshold:.2f}")
            self.update_inference_parameters()

        def update_max_det(value):
            self.max_det = value
            max_det_value.setText(f"{self.max_det}")
            self.update_inference_parameters()

        def update_resolution(value):
            self.resolution = value
            resolution_value.setText(f"{self.resolution}")
            self.update_inference_parameters()

        def update_tta(state):
            self.use_tta = state == Qt.Checked
            self.update_inference_parameters()

        conf_slider.valueChanged.connect(update_conf)
        iou_slider.valueChanged.connect(update_iou)
        max_det_slider.valueChanged.connect(update_max_det)
        resolution_slider.valueChanged.connect(update_resolution)
        tta_checkbox.stateChanged.connect(update_tta)

        apply_button = QPushButton("应用")
        apply_button.clicked.connect(self.apply_parameters)
        layout.addWidget(apply_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def apply_parameters(self):
        self.update_inference_parameters()
        self.restart_inference()

    def show_display_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("调整显示参数")
        layout = QVBoxLayout()

        # 最大边框数量
        max_boxes_layout = QHBoxLayout()
        max_boxes_label = QLabel("最大边界框数量:")
        max_boxes_slider = QSlider(Qt.Horizontal)
        max_boxes_slider.setRange(1, 100)
        max_boxes_slider.setValue(self.max_boxes)
        max_boxes_value = QLabel(f"{self.max_boxes}")
        max_boxes_layout.addWidget(max_boxes_label)
        max_boxes_layout.addWidget(max_boxes_slider)
        max_boxes_layout.addWidget(max_boxes_value)
        layout.addLayout(max_boxes_layout)

        # 重叠阈值
        overlap_layout = QHBoxLayout()
        overlap_label = QLabel("重叠阈值:")
        overlap_slider = QSlider(Qt.Horizontal)
        overlap_slider.setRange(0, 100)
        overlap_slider.setValue(int(self.overlap_threshold * 100))
        overlap_value = QLabel(f"{self.overlap_threshold:.2f}")
        overlap_layout.addWidget(overlap_label)
        overlap_layout.addWidget(overlap_slider)
        overlap_layout.addWidget(overlap_value)
        layout.addLayout(overlap_layout)

        def update_max_boxes(value):
            self.max_boxes = value
            max_boxes_value.setText(f"{self.max_boxes}")

        def update_overlap(value):
            self.overlap_threshold = value / 100
            overlap_value.setText(f"{self.overlap_threshold:.2f}")

        max_boxes_slider.valueChanged.connect(update_max_boxes)
        overlap_slider.valueChanged.connect(update_overlap)

        dialog.setLayout(layout)
        dialog.exec_()

    def show_tracking_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("目标追踪设置")
        layout = QVBoxLayout()

        # 轨迹线条长度
        trajectory_length_layout = QHBoxLayout()
        trajectory_length_label = QLabel("最大轨迹长度:")
        trajectory_length_spinbox = QSpinBox()
        trajectory_length_spinbox.setRange(1, 100)
        trajectory_length_spinbox.setValue(self.max_trajectory_length)
        trajectory_length_layout.addWidget(trajectory_length_label)
        trajectory_length_layout.addWidget(trajectory_length_spinbox)
        layout.addLayout(trajectory_length_layout)

        # 目标存续时间
        max_age_layout = QHBoxLayout()
        max_age_label = QLabel("最大存续时间(帧):")
        max_age_spinbox = QSpinBox()
        max_age_spinbox.setRange(1, 100)
        max_age_spinbox.setValue(self.max_age)
        max_age_layout.addWidget(max_age_label)
        max_age_layout.addWidget(max_age_spinbox)
        layout.addLayout(max_age_layout)

        # 最小命中次数
        min_hits_layout = QHBoxLayout()
        min_hits_label = QLabel("最小命中次数:")
        min_hits_spinbox = QSpinBox()
        min_hits_spinbox.setRange(1, 10)
        min_hits_spinbox.setValue(self.min_hits)
        min_hits_layout.addWidget(min_hits_label)
        min_hits_layout.addWidget(min_hits_spinbox)
        layout.addLayout(min_hits_layout)

        # IOU阈值
        iou_threshold_layout = QHBoxLayout()
        iou_threshold_label = QLabel("IOU阈值:")
        iou_threshold_slider = QSlider(Qt.Horizontal)
        iou_threshold_slider.setRange(0, 100)
        iou_threshold_slider.setValue(int(self.iou_threshold * 100))
        iou_threshold_value = QLabel(f"{self.iou_threshold:.2f}")
        iou_threshold_layout.addWidget(iou_threshold_label)
        iou_threshold_layout.addWidget(iou_threshold_slider)
        iou_threshold_layout.addWidget(iou_threshold_value)
        layout.addLayout(iou_threshold_layout)

        def update_iou_threshold(value):
            self.iou_threshold = value / 100
            iou_threshold_value.setText(f"{self.iou_threshold:.2f}")

        iou_threshold_slider.valueChanged.connect(update_iou_threshold)

        # 确定按钮
        apply_button = QPushButton("应用")
        apply_button.clicked.connect(lambda: self.apply_tracking_settings(
            trajectory_length_spinbox.value(),
            max_age_spinbox.value(),
            min_hits_spinbox.value(),
            self.iou_threshold
        ))
        layout.addWidget(apply_button)

        dialog.setLayout(layout)
        dialog.exec_()

    def apply_tracking_settings(self, max_trajectory_length, max_age, min_hits, iou_threshold):
        self.max_trajectory_length = max_trajectory_length
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # 更新 ObjectTracker
        self.object_tracker = ObjectTracker(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)
        
        # 如果正在进行推理,重新启动推理过程
        if self.is_inferencing:
            self.stop_inference()
            self.start_inference()
        
        QMessageBox.information(self, "设置已更新", "目标追踪设置已更新")

    def check_gpu(self):
        if self.gpu_checkbox.isChecked() and torch.cuda.is_available():
            self.device = torch.device("cuda")
            QMessageBox.information(self, "GPU状态", "GPU已启用并正在使用。")
        else:
            self.device = torch.device("cpu")
            QMessageBox.warning(self, "GPU状态", "GPU未启用或不可用，使用CPU进行推。")
        return self.device

    def populate_model_combo(self):
        model_categories = {
            "目标检测": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
            "实例分割": ["yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"],
            "姿态估计": ["yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose"],
            "分类": ["yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls"]
        }

        for category, models in model_categories.items():
            self.model_combo.addItem(category)
            for model in models:
                self.model_combo.addItem(f"  {model}", model)

        self.model_combo.currentIndexChanged.connect(self.on_model_selected)

    def on_model_selected(self, index):
        model_name = self.model_combo.itemData(index)
        if model_name:
            self.download_and_load_model(model_name)

    def download_and_load_model(self, model_name):
        base_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
        model_url = f"{base_url}{model_name}.pt"
        model_path = os.path.join(self.models_dir, f"{model_name}.pt")

        if not os.path.exists(model_path):
            reply = QMessageBox.question(self, "下载模型", f"模型 {model_name} 不存在，是否下载？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                success = self.download_model(model_url, model_path)
                if not success:
                    QMessageBox.critical(self, "错误", f"模型 {model_name} 下载失败。请检查网络连接或稍后重试。")
                    return
            else:
                return

        # 添加文件完整性检查
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            if file_size < 1024 * 1024:  # 假设模型文件至少应该有1MB
                QMessageBox.critical(self, "错误", f"模型文件 {model_name} 大小异常（{file_size}字节），可能下载不完整。请尝试重新下载。")
                return

        self.load_model(model_path)

    def download_model(self, url, path):
        try:
            # 设置重试策略
            session = requests.Session()
            retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retries))

            # 发送 HEAD 请求以获取文件大小
            response = session.head(url, allow_redirects=True, timeout=10)
            total_size = int(response.headers.get('content-length', 0))

            # 发 GET 请求并下载文件
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            self.progress_dialog = QProgressDialog("下载模型...", "取消", 0, 100, self)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setAutoClose(True)
            self.progress_dialog.setAutoReset(True)
            self.progress_dialog.show()

            downloaded_size = 0
            start_time = time.time()
            with open(path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        progress = int(downloaded_size * 100 / total_size) if total_size > 0 else 0
                        self.progress_dialog.setValue(progress)
                        QApplication.processEvents()

                    if self.progress_dialog.wasCanceled():
                        file.close()
                        os.remove(path)
                        return False

                    # 检查是否超时（5分钟）
                    if time.time() - start_time > 300:
                        raise TimeoutError("下载超时")

            self.progress_dialog.close()

            # 验证下载的文件大小
            if total_size > 0:
                actual_size = os.path.getsize(path)
                if actual_size != total_size:
                    raise Exception(f"下载文件大不正确。预：{total_size}字节，实际：{actual_size}字节")

            print(f"Model downloaded successfully to {path}")
            return True
        except (RequestException, TimeoutError) as e:
            error_msg = f"Error downloading model: {str(e)}\n"
            error_msg += f"URL: {url}\n"
            error_msg += f"File path: {path}\n"
            print(error_msg)
            QMessageBox.critical(self, "下载错误", error_msg)
            if os.path.exists(path):
                os.remove(path)  # 删除不完整的文件
            return False

    def load_local_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "选择YOLO模型文件", self.models_dir, "Model Files (*.pt)")
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        try:
            print(f"尝试加载模型: {model_path}")
            logger.info(f"尝试加载模型: {model_path}")
            self.device = self.check_gpu()
            print(f"使用设备: {self.device}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            file_size = os.path.getsize(model_path)
            print(f"模型文件大小: {file_size} 字节")
            
            if file_size == 0:
                raise ValueError("模型文件大小为0，可能下载不完整")
            
            print("开始加载模型...")
            self.model = YOLO(model_path)
            print(f"模型加载成功: {self.model}")
            
            self.model = self.model.to(self.device)
            print(f"模型已移动到设备: {self.device}")
            
            self.class_names = self.model.names
            print(f"类别名称: {self.class_names}")
            
            # 确定模型类型
            if 'seg' in model_path:
                self.current_model_type = 'segmentation'
                self.tracking_checkbox.setEnabled(True)  # 启用追踪选项
            elif 'pose' in model_path:
                self.current_model_type = 'pose'
                self.tracking_checkbox.setEnabled(False)  # 禁用追踪选项
                self.tracking_checkbox.setChecked(False)  # 取消选中
            elif 'cls' in model_path:
                self.current_model_type = 'classification'
                self.tracking_checkbox.setEnabled(False)  # 禁用追踪选项
                self.tracking_checkbox.setChecked(False)  # 取消选中
            else:
                self.current_model_type = 'detection'
                self.tracking_checkbox.setEnabled(True)  # 启用追踪选项
            
            print(f"当前模型类型: {self.current_model_type}")
            
            QMessageBox.information(self, "成功", f"YOLO11模型加载成功！类型: {self.current_model_type}, 使用设备: {self.device}")
            self.start_button.setEnabled(True)
        except Exception as e:
            error_msg = f"加载YOLO11模型时出错：{str(e)}\n"
            error_msg += f"模型路径: {model_path}\n"
            error_msg += f"模型文件夹: {self.models_dir}\n"
            error_msg += f"Python版本: {sys.version}\n"
            error_msg += f"PyTorch版本: {torch.__version__}\n"
            error_msg += f"CUDA是否可用: {torch.cuda.is_available()}\n"
            if torch.cuda.is_available():
                error_msg += f"CUDA版本: {torch.version.cuda}\n"
            
            print(error_msg)
            logger.error(error_msg)
            QMessageBox.critical(self, "错误", error_msg)

    def start_camera(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.start_button.setEnabled(True)
            self.start_button.setText("开始推理")  # 重置按钮文本
            
            # 读取第一帧并显示
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_captured_frame(frame_rgb)
            else:
                raise Exception("无法读取摄像头帧")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动摄像时出错：{str(e)}")

    def load_video(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
        try:
            self.reset_state()  # 重置状态
            file_name, _ = QFileDialog.getOpenFileName(
                self, 
                "选择视频或图片文件", 
                "", 
                "所有支持的文件 (*.mp4 *.avi *.jpg *.jpeg *.png);;视频文件 (*.mp4 *.avi);;图片文件 (*.jpg *.jpeg *.png)"
            )
            if file_name:
                file_extension = os.path.splitext(file_name)[1].lower()
                if file_extension in ['.jpg', '.jpeg', '.png']:
                    self.load_image(file_name)
                elif file_extension in ['.mp4', '.avi']:
                    self.load_video_file(file_name)
                else:
                    raise ValueError("不支持的文件格式")
            self.start_button.setEnabled(True)  # 确保按钮被启用
            self.start_button.setText("开始推理")  # 重置按钮文本
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载文件时出错：{str(e)}")

    def load_image(self, file_name):
        self.reset_state()  # 重置状态
        self.is_image = True
        self.preview_image = cv2.imread(file_name)
        self.preview_image = cv2.cvtColor(self.preview_image, cv2.COLOR_BGR2RGB)
        self.display_preview(self.preview_image)
        self.start_button.setEnabled(True)

    def load_video_file(self, file_name):
        self.reset_state()  # 重置状态
        self.cap = cv2.VideoCapture(file_name)
        if not self.cap.isOpened():
            raise Exception("无法打开视频文件")
        self.is_image = False
        self.current_video_path = file_name  # 添加这行
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_preview(frame_rgb)
        else:
            raise Exception("无法读取视频帧")
        self.start_button.setEnabled(True)

    def display_preview(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_inference(self):
        if not self.is_inferencing:
            try:
                logger.info("Starting inference")
                self.is_inferencing = True
                self.is_capturing = False

                if self.is_image:
                    results = self.model(self.preview_image, conf=self.confidence_threshold, 
                                         iou=self.iou_threshold, max_det=self.max_det)
                    self.process_inference_result(self.preview_image, results)
                    self.is_inferencing = False
                    self.start_button.setText("开始推理")
                else:
                    if hasattr(self, 'capture_thread'):
                        self.capture_thread.join()
                    self.inference_thread = InferenceThread(self.model, self.device)
                    self.inference_thread.update_parameters(self.confidence_threshold, self.iou_threshold, self.max_det)
                    self.inference_thread.inference_finished.connect(self.process_inference_result)
                    self.inference_thread.start()
                    self.start_button.setText("停止推理")
                    self.timer.start(int(1000 / self.fps))

                logger.info("Inference started successfully")
            except Exception as e:
                logger.error(f"Error starting inference: {str(e)}")
                QMessageBox.critical(self, "错误", f"启动推理时出错：{str(e)}")
                self.is_inferencing = False
        else:
            self.stop_inference()

    def stop_inference(self):
        try:
            logger.info("Stopping inference")
            self.is_inferencing = False
            self.timer.stop()
            if self.inference_thread:
                self.inference_thread.stop()
                self.inference_thread.wait()
                self.inference_thread = None
            self.start_button.setText("开始推理")
            if not self.is_image:
                self.is_capturing = True
                self.capture_thread = threading.Thread(target=self.capture_loop)
                self.capture_thread.start()
            logger.info("Inference stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping inference: {str(e)}")
            QMessageBox.critical(self, "错误", f"停止推理时出错：{str(e)}")

    def capture_loop(self):
        logger.info("Capture loop started")
        while self.is_capturing and not self.is_inferencing:
            try:
                if self.cap is not None and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_buffer.append(frame_rgb)
                        self.display_captured_frame(frame_rgb)
                    else:
                        logger.warning("Failed to read frame from camera")
                else:
                    logger.warning("Camera is not opened")
                    break
            except Exception as e:
                logger.error(f"Error in capture loop: {str(e)}")
            time.sleep(1 / self.fps)
        logger.info("Capture loop ended")

    def update_screen_capture(self):
        if self.capture_window and self.capture_window.isVisible() and not self.capture_window.initial_selection:
            try:
                screen = QApplication.primaryScreen()
                if self.capture_rect:
                    self.captured_pixmap = screen.grabWindow(0, self.capture_rect.x(), self.capture_rect.y(), 
                                                             self.capture_rect.width(), self.capture_rect.height())
                    frame = self.pixmap_to_cv2(self.captured_pixmap)
                    
                    if frame is not None and frame.size > 0:
                        self.frame_buffer.append(frame)
                        if self.is_inferencing and self.inference_thread:
                            self.inference_thread.update_frame(frame)
                    else:
                        logger.warning("Captured frame is empty or invalid")
                else:
                    logger.warning("Capture rectangle is not set")
            except Exception as e:
                logger.error(f"Error in update_screen_capture: {str(e)}")
        else:
            logger.warning("Capture window is not visible, not initialized, or initial selection is not complete")

    def stop(self):
        """停止所有捕获和推理"""
        logger.info("Stopping all capture and inference...")
        
        # 停止推理
        self.is_inferencing = False
        
        # 停止所有定时器
        self.timer.stop()
        self.capture_timer.stop()
        self.display_timer.stop()
        
        # 停止推理线程
        if self.inference_thread:
            try:
                self.inference_thread.stop()
                self.inference_thread.wait()
                self.inference_thread = None
            except Exception as e:
                logger.error(f"Error stopping inference thread: {str(e)}")
        
        # 停止窗口捕获线程
        if self.window_capture_thread:
            try:
                self.window_capture_thread.stop()
                self.window_capture_thread.wait()
                self.window_capture_thread = None
            except Exception as e:
                logger.error(f"Error stopping window capture thread: {str(e)}")
        
        # 停止摄像头捕获
        if self.cap is not None:
            try:
                self.cap.release()
                self.cap = None
            except Exception as e:
                logger.error(f"Error releasing camera: {str(e)}")
        
        # 关闭屏幕捕获窗口
        if self.capture_window:
            try:
                self.capture_window.hide()
            except Exception as e:
                logger.error(f"Error hiding capture window: {str(e)}")
        
        # 重置按钮状态
        self.start_button.setText("开始推理")
        self.start_button.setEnabled(True)
        
        logger.info("All capture and inference stopped successfully")

    def reset_state(self):
        self.is_image = False
        self.is_inferencing = False
        self.is_capturing = False
        self.preview_image = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.frame_buffer.clear()
        self.result_buffer.clear()
        self.start_button.setText("开始推理")
        self.image_label.clear()
        self.last_annotated_frame = None
        self.save_result_button.setEnabled(False)
        self.counting_line = None
        self.object_tracks.clear()
        self.count_up = 0
        self.count_down = 0
        self.track_id = 0
        self.class_counts = {}
        self.class_line_counts = {}
        self.max_class_counts = {}
        self.export_stats_button.setEnabled(False)

    def adjust_screen_capture(self):
        """调整屏幕捕获区域"""
        try:
            # 停止之前的捕获
            self.stop()
            
            # 停止进程捕获线程（如果存在）
            if self.window_capture_thread:
                self.window_capture_thread.stop()
                self.window_capture_thread.wait()
                self.window_capture_thread = None
            
            # 创建或显示屏幕捕获窗口
            if not self.capture_window:
                self.capture_window = ResizableTransparentWindow()
                self.capture_window.capture_updated.connect(self.on_capture_updated)
            
            self.capture_window.initial_selection = True
            self.capture_window.setGeometry(QApplication.primaryScreen().geometry())
            self.capture_window.show()
            
            # 启动屏幕捕获定时器
            self.capture_timer.start(33)
            self.capture_rect = None
            self.start_button.setEnabled(True)
            self.start_button.setText("开始推理")
            
        except Exception as e:
            logger.error(f"调整屏幕捕获时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"调整屏幕捕获时出错: {str(e)}")

    def on_capture_updated(self, geometry):
        self.capture_rect = geometry
        self.update_screen_capture()
        self.display_captured_frame(self.frame_buffer[-1] if self.frame_buffer else None)
        self.start_button.setEnabled(True)
        self.start_button.setText("开始推理")

    def pixmap_to_cv2(self, pixmap):
        image = pixmap.toImage()
        s = image.bits().asstring(image.byteCount())
        arr = np.frombuffer(s, dtype=np.uint8).reshape((image.height(), image.width(), 4))
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)

    def display_captured_frame(self, frame):
        if frame is not None:
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            print("Error: No frame to display")

    def get_real_position(self, pos):
        screen = QApplication.primaryScreen()
        dpr = screen.devicePixelRatio()
        return QPoint(int(pos.x() * dpr), int(pos.y() * dpr))

    def non_max_suppression(self, boxes, scores, threshold):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return keep

    def generate_colors(self, num_classes):
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        return colors

    def get_class_color(self, class_id):
        if class_id not in self.class_colors:
            if not self.class_colors:
                colors = self.generate_colors(len(self.class_names))
                self.class_colors = {i: color for i, color in enumerate(colors)}
            else:
                self.class_colors[class_id] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.class_colors[class_id]

    def start_detection(self):
        if self.is_image:
            img = np.array(self.preview_image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            results = self.model(img)
            self.display_results(img, results)
        else:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    results = self.model(frame)
                    self.display_results(frame, results)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            self.cap.release()
            cv2.destroyAllWindows()

    def display_results(self, img, results):
        annotated_frame = self.annotate_frame(img, results)
        if self.is_image:
            cv2.imshow('Object Detection Result', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imshow('Object Detection Result', annotated_frame)

    def update_frame(self):
        if not self.is_image and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_buffer.append(frame_rgb)
                if self.is_inferencing and self.inference_thread:
                    self.inference_thread.update_frame(frame_rgb)
            else:
                self.stop()

    def process_inference_result(self, original_frame, results):
        logger.info(f"Results type in process_inference_result: {type(results)}")
        
        detections = []
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])
                detections.append([x1, y1, x2, y2, conf, cls])
        
        if self.tracking_checkbox.isChecked():
            self.object_tracker.update(detections)
            # 传递显示框的状态到draw_tracks方法
            tracked_frame = self.object_tracker.draw_tracks(
                original_frame.copy(), 
                max_trajectory_length=self.max_trajectory_length,
                show_boxes=self.show_boxes_checkbox.isChecked()
            )
        else:
            tracked_frame = original_frame.copy()
        
        self.last_annotated_frame = self.annotate_frame(tracked_frame, results)
        if self.counting_line:
            self.count_objects(results[0])
        self.count_classes(results[0])
        self.display_frame(self.last_annotated_frame)
        self.save_result_button.setEnabled(True)
        self.export_stats_button.setEnabled(True)
        if self.is_image:
            self.is_inferencing = False
            self.start_button.setText("开始推理")

    def count_classes(self, results):
        current_counts = {}
        if hasattr(results, 'boxes') and results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls.cpu().numpy()[0])
                class_name = self.class_names[cls] if self.class_names else f"Class {cls}"
                if class_name in current_counts:
                    current_counts[class_name] += 1
                else:
                    current_counts[class_name] = 1

        for class_name, count in current_counts.items():
            if class_name not in self.max_class_counts or count > self.max_class_counts[class_name]:
                self.max_class_counts[class_name] = count

        self.class_counts = current_counts
        logger.info(f"Current class counts: {self.class_counts}")
        logger.info(f"Max class counts: {self.max_class_counts}")

    def update_display(self):
        if self.is_inferencing and self.result_buffer:
            original_frame, results = self.result_buffer[-1]
            annotated_frame = self.annotate_frame(original_frame, results)
            self.display_frame(annotated_frame)
        elif not self.is_inferencing and self.frame_buffer:
            self.display_frame(self.frame_buffer[-1])

    def annotate_frame(self, frame, results):
        annotated_frame = frame.copy()
        if self.current_model_type == 'detection':
            annotated_frame = self.display_detection_results(annotated_frame, results[0])
        elif self.current_model_type == 'segmentation':
            annotated_frame = self.display_segmentation_results(annotated_frame, results[0])
        elif self.current_model_type == 'pose':
            annotated_frame = self.display_pose_results(annotated_frame, results[0])
        elif self.current_model_type == 'classification':
            annotated_frame = self.display_classification_results(annotated_frame, results)
        return annotated_frame

    def display_frame(self, frame):
        if frame is not None and frame.size > 0:
            try:
                # 确保帧数据有效
                if len(frame.shape) == 3 and frame.shape[2] >= 3:
                    height, width, channel = frame.shape
                    bytes_per_line = 3 * width
                    
                    # 确保使用RGB格式
                    if channel == 4:  # 如果是RGBA格式
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    
                    # 创建QImage
                    q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    
                    # 获取标签大小并保持纵横比
                    label_size = self.image_label.size()
                    scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    
                    # 计算缩放比例
                    self.image_scale = scaled_pixmap.width() / width
                    
                    # 创建新的pixmap用于绘制
                    result_pixmap = QPixmap(label_size)
                    result_pixmap.fill(Qt.transparent)
                    
                    # 创建painter
                    painter = QPainter(result_pixmap)
                    
                    # 计算居中位置
                    x = (label_size.width() - scaled_pixmap.width()) // 2
                    y = (label_size.height() - scaled_pixmap.height()) // 2
                    
                    # 绘制图像
                    painter.drawPixmap(x, y, scaled_pixmap)
                    
                    # 如果有计数线，绘制它
                    if self.counting_line and len(self.counting_line) == 4:
                        painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.SolidLine))
                        scaled_line = [int(p * self.image_scale) for p in self.counting_line]
                        painter.drawLine(scaled_line[0], scaled_line[1], scaled_line[2], scaled_line[3])
                    
                    # 如果显示统计数据
                    if self.show_statistics:
                        painter.setPen(QPen(QColor(0, 255, 0)))
                        painter.setFont(QFont("Arial", 12))
                        
                        # 绘制计数数据
                        y_offset = 30
                        painter.drawText(10, y_offset, f"上行: {self.count_up}")
                        y_offset += 30
                        painter.drawText(10, y_offset, f"下行: {self.count_down}")
                        
                        # 绘制类别计数
                        for class_name, count in self.class_counts.items():
                            y_offset += 30
                            max_count = self.max_class_counts.get(class_name, 0)
                            painter.drawText(10, y_offset, f"{class_name}: {count}/{max_count}")
                        
                        # 绘制经过计数线的次数
                        for class_name, count in self.class_line_counts.items():
                            y_offset += 30
                            painter.drawText(10, y_offset, f"{class_name} 经过: {count}")
                    
                    painter.end()
                    
                    # 显示结果
                    self.image_label.setPixmap(result_pixmap)
                    
                else:
                    logger.warning(f"Invalid frame shape in display_frame: {frame.shape}")
            except Exception as e:
                logger.error(f"Error in display_frame: {str(e)}")
        else:
            logger.warning("Attempted to display None or empty frame")

    def display_detection_results(self, img, results):
        if not hasattr(results, 'boxes') or results.boxes is None:
            return img

        boxes = results.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf.cpu().numpy()[0]
            cls = int(box.cls.cpu().numpy()[0])
            
            color = self.get_class_color(cls)
            # 只在复选框选中时绘制边界框
            if self.show_boxes_checkbox.isChecked():
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                class_name = self.class_names[cls] if self.class_names else f"Class {cls}"
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

    def display_segmentation_results(self, img, results):
        if not hasattr(results, 'masks') or results.masks is None:
            print("No masks found in segmentation results")
            return img

        try:
            # 将图像转换为 PyTorch tensor 并移至 GPU
            img_tensor = torch.from_numpy(img.copy()).to(self.device)
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

            masks = results.masks
            boxes = results.boxes

            # 创建一个与原图大小相同的掩码张量，初始值为0
            final_mask = torch.zeros((img.shape[0], img.shape[1]), device=self.device, dtype=torch.float32)
            colored_overlay = torch.zeros((img.shape[0], img.shape[1], 3), device=self.device, dtype=torch.float32)

            for mask, box in zip(masks, boxes):
                # 获取掩码数据并调整大小
                mask_tensor = mask.data[0].to(self.device)  # [H, W]
                if mask_tensor.shape[-2:] != (img.shape[0], img.shape[1]):
                    mask_tensor = F.interpolate(
                        mask_tensor.unsqueeze(0).unsqueeze(0),
                        size=(img.shape[0], img.shape[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                cls = int(box.cls.item())
                color = self.get_class_color(cls)
                
                # 将掩码转换为二值掩码
                binary_mask = (mask_tensor > 0.5).float()
                
                # 更新最终掩码和颜色叠加层
                final_mask = torch.max(final_mask, binary_mask)
                
                # 为每个类别添加颜色
                for c in range(3):  # RGB通道
                    colored_overlay[..., c] += binary_mask * color[c]

                # 在原图上绘制边界框和标签
                if self.show_boxes_checkbox.isChecked():
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0]
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    class_name = self.class_names[cls] if self.class_names else f"Class {cls}"
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(img, label, (int(x1), int(y1) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 应用掩码和颜色
            if torch.any(final_mask > 0):
                # 将图像转换为浮点数进行混合
                img_float = img_tensor.squeeze(0).permute(1, 2, 0).float()
                
                # 扩展掩码维度以匹配图像通道
                final_mask = final_mask.unsqueeze(-1).expand(-1, -1, 3)
                
                # 在有掩码的区域进行混合
                alpha = 0.5  # 透明度
                blended = torch.where(
                    final_mask > 0,
                    img_float * (1 - alpha) + colored_overlay * alpha,
                    img_float
                )
                
                # 确保值在有效范围内并转回uint8
                result = torch.clamp(blended, 0, 255).cpu().numpy().astype(np.uint8)
                return result

            return img

        except Exception as e:
            logger.error(f"Error in display_segmentation_results: {str(e)}")
            logger.exception(e)  # 打印完整的错误堆栈
            return img

    def display_pose_results(self, img, results):
        if not hasattr(results, 'keypoints') or results.keypoints is None:
            return img

        keypoints = results.keypoints
        for kpts in keypoints:
            kpts = kpts.cpu().numpy()
            for i in range(kpts.shape[0]):
                x, y = kpts[i, 0], kpts[i, 1]
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

        return img

    def display_classification_results(self, img, results):
        if not hasattr(results, 'probs') or results.probs is None:
            return img

        probs = results.probs.cpu().numpy()
        class_id = np.argmax(probs)
        class_name = self.class_names[class_id] if self.class_names else f"Class {class_id}"
        label = f"{class_name}: {probs[class_id]:.2f}"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

    def display_pose_results(self, img, results):
        if not hasattr(results, 'keypoints') or results.keypoints is None:
            print("No keypoints found in pose results")
            return img

        keypoints = results.keypoints
        boxes = results.boxes

        # 定义不同身体部位的颜色
        colors = {
            'head': (255, 0, 0),    # 蓝色
            'body': (0, 255, 0),    # 绿色
            'arms': (255, 165, 0),  # 橙色
            'legs': (255, 0, 255)   # 紫色
        }

        for person_keypoints, box in zip(keypoints, boxes):
            kpts = person_keypoints.data[0]
            for kpt in kpts:
                x, y, conf = kpt
                if conf > 0.5:
                    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

            connections = [
                ((0, 1), 'head'), ((0, 2), 'head'), ((1, 3), 'head'), ((2, 4), 'head'),
                ((0, 5), 'body'), ((0, 6), 'body'),
                ((5, 6), 'body'), ((5, 11), 'body'), ((6, 12), 'body'), ((11, 12), 'body'),
                ((5, 7), 'arms'), ((7, 9), 'arms'), ((6, 8), 'arms'), ((8, 10), 'arms'),
                ((11, 13), 'legs'), ((13, 15), 'legs'), ((12, 14), 'legs'), ((14, 16), 'legs')
            ]
            for (connection, body_part) in connections:
                pt1, pt2 = kpts[connection[0]], kpts[connection[1]]
                if pt1[2] > 0.5 and pt2[2] > 0.5:
                    cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), colors[body_part], 2)

            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Person: {conf:.2f}"
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

    def display_classification_results(self, img, results):
        if not hasattr(results[0], 'probs'):
            print("No probabilities found in classification results")
            return img

        probs = results[0].probs
        if probs is None:
            print("Probabilities are None")
            return img

        top_indices = probs.top5

        for i, idx in enumerate(top_indices):
            class_name = self.class_names[idx] if self.class_names else f"Class {idx}"
            prob = probs.top5conf[i].item()
            label = f"{class_name}: {prob:.2f}"
            cv2.putText(img, label, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

    def closeEvent(self, event):
        self.stop()
        if self.capture_window:
            self.capture_window.close()
        event.accept()

    def update_inference_parameters(self):
        if self.model:
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            self.model.max_det = self.max_det
            # 更新分辨率和TTA设置
            self.model.imgsz = self.resolution
            self.model.augment = self.use_tta
        if self.inference_thread:
            self.restart_inference()
        logger.info(f"Updated inference parameters: conf={self.confidence_threshold}, iou={self.iou_threshold}, max_det={self.max_det}, resolution={self.resolution}, use_tta={self.use_tta}")

    def restart_inference(self):
        if self.inference_thread:
            self.stop_inference()
        self.start_inference()

    def stop_inference(self):
        if self.inference_thread:
            self.inference_thread.stop()
            self.inference_thread.wait()
            self.inference_thread = None

    def save_result(self):
        if self.last_annotated_frame is None:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return

        if self.is_image:
            file_name, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "图片文件 (*.png *.jpg *.jpeg)")
            if file_name:
                cv2.imwrite(file_name, cv2.cvtColor(self.last_annotated_frame, cv2.COLOR_RGB2BGR))
                QMessageBox.information(self, "成功", f"图片已保存到 {file_name}")
        else:
            file_name, _ = QFileDialog.getSaveFileName(self, "保存视频", "", "视频文件 (*.mp4)")
            if file_name:
                temp_video_path = "temp_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                height, width = self.last_annotated_frame.shape[:2]
                out = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (width, height))
                
                # 保存当前帧
                out.write(cv2.cvtColor(self.last_annotated_frame, cv2.COLOR_RGB2BGR))
                
                # 如果是视频，继续处理剩余帧
                if self.cap is not None and self.cap.isOpened():
                    original_position = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到视频开始
                    
                    while True:
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.model(frame_rgb)
                        annotated_frame = self.annotate_frame(frame_rgb, results)
                        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                    
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_position)  # 恢复原始位置
                
                out.release()

                # 使用 moviepy 处理音频
                try:
                    from moviepy.editor import VideoFileClip, AudioFileClip
                    
                    original_clip = VideoFileClip(self.current_video_path)
                    new_clip = VideoFileClip(temp_video_path)
                    
                    # 如果原始视频有音频，则添加到新视频
                    if original_clip.audio:
                        new_clip = new_clip.set_audio(original_clip.audio)
                    
                    new_clip.write_videofile(file_name, codec="libx264", audio_codec="aac")
                    
                    original_clip.close()
                    new_clip.close()
                    
                    # 删除临时文件
                    os.remove(temp_video_path)
                    
                    QMessageBox.information(self, "成功", f"视频已保存到 {file_name}")
                except Exception as e:
                    logger.error(f"Error processing video with audio: {str(e)}")
                    QMessageBox.warning(self, "警告", f"保存视频时出错：{str(e)}")

        logger.info(f"Result saved to {file_name}")

    def set_counting_line(self):
        """设置计数线"""
        try:
            if self.last_annotated_frame is not None or self.frame_buffer:
                self.counting_line = None
                self.is_setting_line = True
                QMessageBox.information(self, "设置计数线", "请在图像上点击并拖动来设置计数线")
            else:
                QMessageBox.warning(self, "警告", "请先加载图像或视频并进行推理")
        except Exception as e:
            logger.error(f"Error in set_counting_line: {str(e)}")
            QMessageBox.critical(self, "错误", f"设置计数线时出错：{str(e)}")

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.is_setting_line:
            # 获取图像标签的位置和大小
            label_rect = self.image_label.geometry()
            # 获取鼠标在窗口中的位置
            mouse_pos = event.pos()
            # 计算鼠标相对于图像标签的位置
            relative_pos = QPoint(mouse_pos.x() - label_rect.x(), mouse_pos.y() - label_rect.y())
            
            # 检查点击是否在图像标签内
            if label_rect.contains(mouse_pos):
                self.origin = relative_pos
                if not self.rubber_band:
                    self.rubber_band = QRubberBand(QRubberBand.Line, self.image_label)
                self.rubber_band.setGeometry(QRect(self.origin, QSize()))
                self.rubber_band.show()

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.is_setting_line and self.rubber_band:
            # 获取图像标签的位置和大小
            label_rect = self.image_label.geometry()
            # 获取鼠标在窗口中的位置
            mouse_pos = event.pos()
            # 计算鼠标相对于图像标签的位置
            relative_pos = QPoint(mouse_pos.x() - label_rect.x(), mouse_pos.y() - label_rect.y())
            
            # 检查移动是否在图像标签内
            if label_rect.contains(mouse_pos):
                self.rubber_band.setGeometry(QRect(self.origin, relative_pos).normalized())

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if self.is_setting_line:
            try:
                if self.rubber_band:
                    self.rubber_band.hide()
                    
                    # 获取图像标签的位置和大小
                    label_rect = self.image_label.geometry()
                    # 获取图像标签中显示的图像的实际大小
                    pixmap = self.image_label.pixmap()
                    if pixmap:
                        pixmap_rect = self.get_scaled_pixmap_rect()
                        
                        # 计算开始和结束点相对于实际显示图像的位置
                        start_pos = QPoint(
                            self.origin.x() - pixmap_rect.x(),
                            self.origin.y() - pixmap_rect.y()
                        )
                        end_pos = QPoint(
                            event.pos().x() - label_rect.x() - pixmap_rect.x(),
                            event.pos().y() - label_rect.y() - pixmap_rect.y()
                        )
                        
                        # 转换坐标到原始图像尺寸
                        self.counting_line = [
                            int(start_pos.x() / self.image_scale),
                            int(start_pos.y() / self.image_scale),
                            int(end_pos.x() / self.image_scale),
                            int(end_pos.y() / self.image_scale)
                        ]
                        
                        self.is_setting_line = False
                        
                        # 重新显示当前帧
                        if self.last_annotated_frame is not None:
                            self.display_frame(self.last_annotated_frame)
                        elif self.frame_buffer:
                            self.display_frame(self.frame_buffer[-1])
                        
                        QMessageBox.information(self, "计数线设置", "计数线已设置完成")
                        logger.info(f"Counting line set: {self.counting_line}")
            except Exception as e:
                logger.error(f"Error in mouseReleaseEvent: {str(e)}")
                QMessageBox.critical(self, "错误", f"结束设置计数线时出错：{str(e)}")

    def get_scaled_pixmap_rect(self):
        """获取实际显示的图像区域"""
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return QRect()
        
        # 获取标签和图像的尺寸
        label_size = self.image_label.size()
        scaled_size = pixmap.size()
        scaled_size.scale(label_size, Qt.KeepAspectRatio)
        
        # 计算图像在标签中的位置（居中显示）
        x = (label_size.width() - scaled_size.width()) // 2
        y = (label_size.height() - scaled_size.height()) // 2
        
        return QRect(x, y, scaled_size.width(), scaled_size.height())

    def count_objects(self, results):
        if self.counting_line is None or len(self.counting_line) != 4:
            return

        if not hasattr(results, 'boxes') or results.boxes is None:
            return

        boxes = results.boxes
        current_tracks = {}

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls.cpu().numpy()[0])
            class_name = self.class_names[cls] if self.class_names else f"Class {cls}"
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            min_distance = float('inf')
            closest_track_id = None

            for track_id, track in self.object_tracks.items():
                if track:#@文抑青年
                    last_x, last_y = track[-1]
                    distance = np.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_track_id = track_id

            if min_distance < 50:  # 假设50像素为阈值
                current_tracks[closest_track_id] = (center_x, center_y, class_name)
                if len(self.object_tracks[closest_track_id]) > 1:
                    prev_point = self.object_tracks[closest_track_id][-1]
                    curr_point = (center_x, center_y)
                    if self.intersect(prev_point, curr_point, 
                                      (self.counting_line[0], self.counting_line[1]),
                                      (self.counting_line[2], self.counting_line[3])):
                        if center_y > prev_point[1]:
                            self.count_down += 1
                        else:
                            self.count_up += 1
                        
                        # 更新类别经过计数线的次数
                        if class_name in self.class_line_counts:
                            self.class_line_counts[class_name] += 1
                        else:
                            self.class_line_counts[class_name] = 1
            else:
                self.track_id += 1
                current_tracks[self.track_id] = (center_x, center_y, class_name)

        # 更新跟踪
        for track_id, (x, y, class_name) in current_tracks.items():
            if track_id in self.object_tracks:
                self.object_tracks[track_id].append((x, y))
            else:
                self.object_tracks[track_id] = [(x, y)]

        # 移除旧的跟踪
        self.object_tracks = {k: v for k, v in self.object_tracks.items() if k in current_tracks}

        # 打印日志以便调试
        logger.info(f"Current counts - Up: {self.count_up}, Down: {self.count_down}")
        logger.info(f"Class line counts: {self.class_line_counts}")

    def intersect(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def process_frame(self, frame):
        # 使用模型进行推理
        results = self.model(frame)
        
        # 获取检测结果
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            
            detections = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])
                detections.append([x1, y1, x2, y2, conf, cls])
            
            # 更新轨迹
            self.object_tracker.update(detections)
            
            for box in boxes:
                cls = int(box.cls.cpu().numpy()[0])
                conf = box.conf.cpu().numpy()[0]
                class_name = self.class_names[cls] if self.class_names else f"Class {cls}"
                
                self.detection_results.append({
                    'class': class_name,
                    'confidence': conf,
                    'timestamp': pd.Timestamp.now()
                })
        
        # 返回处理后的帧（如果需要的话）
        return self.annotate_frame(frame, results)

    def export_results(self):
        if not self.detection_results:
            print("没有可导出的检测结果")
            return
        
        df = pd.DataFrame(self.detection_results)
        
        # 计算每个类别的检测次数
        class_counts = df['class'].value_counts()
        
        # 计每个类别的平均置信度
        avg_confidence = df.groupby('class')['confidence'].mean()
        
        # 合并结果
        summary = pd.DataFrame({
            'Count': class_counts,
            'Average Confidence': avg_confidence
        }).reset_index()
        summary.columns = ['Class', 'Count', 'Average Confidence']
        
        # 导出到Excel文件
        filename = f"detection_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with pd.ExcelWriter(filename) as writer:
            summary.to_excel(writer, sheet_name='Summary', index=False)
            df.to_excel(writer, sheet_name='Raw Data', index=False)
        
        print(f"结果已导出到 {filename}")

    def export_statistics(self):
        try:
            if not self.class_counts and not self.max_class_counts and not self.class_line_counts:
                QMessageBox.warning(self, "警告", "没有可导出的统计数据")
                return

            # 创建数据框
            data = {
                '类别': [],
                '当前计数': [],
                '最大计数': [],
                '经过计数线次数': []
            }

            # 合并所有类别
            all_classes = set(list(self.class_counts.keys()) + 
                              list(self.max_class_counts.keys()) + 
                              list(self.class_line_counts.keys()))

            for cls in all_classes:
                data['类别'].append(cls)
                data['当前计数'].append(self.class_counts.get(cls, 0))
                data['最大计数'].append(self.max_class_counts.get(cls, 0))
                data['经过计数线次数'].append(self.class_line_counts.get(cls, 0))

            df = pd.DataFrame(data)

            # 添加总计行
            total_row = pd.DataFrame({
                '类别': ['总计'],
                '当前计数': [df['当前计数'].sum()],
                '最大计数': [df['最大计数'].sum()],
                '经过计数线次数': [df['经过计数线次数'].sum()]
            })
            df = pd.concat([df, total_row], ignore_index=True)

            # 添加上行/下行计数
            up_down_df = pd.DataFrame({
                '类别': ['上行计数', '下行计数'],
                '当前计数': [self.count_up, self.count_down],
                '最大计数': [self.count_up, self.count_down],
                '经过计数线次数': [self.count_up, self.count_down]
            })
            df = pd.concat([df, up_down_df], ignore_index=True)

            # 导出到文件
            file_name, _ = QFileDialog.getSaveFileName(self, "保存统计数据", "", "Excel文件 (*.xlsx);;CSV文件 (*.csv)")
            if file_name:
                if file_name.endswith('.xlsx'):
                    try:
                        df.to_excel(file_name, index=False, sheet_name='统计数据')
                    except ImportError:
                        QMessageBox.warning(self, "警告", "无法导出为Excel格式，将保存为CSV格式")
                        file_name = file_name[:-5] + '.csv'
                        df.to_csv(file_name, index=False, encoding='utf-8-sig')
                elif file_name.endswith('.csv'):
                    df.to_csv(file_name, index=False, encoding='utf-8-sig')
                else:
                    file_name += '.csv'
                    df.to_csv(file_name, index=False, encoding='utf-8-sig')
                
                QMessageBox.information(self, "成功", f"统计数据已保存到 {file_name}")
                logger.info(f"Statistics exported to {file_name}")
        except Exception as e:
            logger.error(f"Error exporting statistics: {str(e)}")
            QMessageBox.critical(self, "错误", f"导出统计数据时出错：{str(e)}")

    def reset_statistics(self):
        """重置所有统计数据"""
        self.class_counts = {}
        self.max_class_counts = {}
        self.class_line_counts = {}
        self.count_up = 0
        self.count_down = 0
        self.track_id = 0
        self.object_tracks.clear()
        QMessageBox.information(self, "成功", "统计数据已重置")
        self.display_frame(self.last_annotated_frame)  

    def toggle_statistics_display(self):
        """切换统计数据的显示状态"""
        self.show_statistics = not self.show_statistics
        if self.show_statistics:
            self.toggle_stats_button.setText("隐藏统计数据")
        else:
            self.toggle_stats_button.setText("显示统计数据")
        self.display_frame(self.last_annotated_frame)  

    def start_window_capture(self):
        """启动进程窗口捕获"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
        
        try:
            # 停止之前的捕获
            self.stop()
            
            # 关闭屏幕捕获窗口（如果存在）
            if self.capture_window:
                self.capture_window.hide()
                self.capture_window = None
            
            # 停止屏幕捕获定时器
            self.capture_timer.stop()
            self.capture_rect = None
            
            # 创建新的窗口捕获线程
            self.window_capture_thread = WindowCaptureThread(self)
            self.window_capture_thread.frame_captured.connect(self.on_window_frame_captured)
            
            # 获取进程列表
            processes = self.window_capture_thread.capturer.list_processes()
            if not processes:
                QMessageBox.warning(self, "警告", "未找到可用进程")
                return
                
            # 创建进程选择对话框
            dialog = QDialog(self)
            dialog.setWindowTitle("选择要捕获的进程")
            layout = QVBoxLayout()
            
            # 添加进程列表
            combo = QComboBox()
            for proc in processes:
                combo.addItem(f"{proc['name']} ({proc['title']}) - PID: {proc['pid']}", proc)
            layout.addWidget(combo)
            
            # 添加确定按钮
            button = QPushButton("确定")
            button.clicked.connect(dialog.accept)
            layout.addWidget(button)
            
            dialog.setLayout(layout)
            
            if dialog.exec_() == QDialog.Accepted:
                selected_proc = combo.currentData()
                if selected_proc:  # 确保选择了有效的进程
                    if self.window_capture_thread.capturer.find_window_by_pid_and_title(
                        selected_proc['pid'], 
                        selected_proc['title']
                    ):
                        self.window_capture_thread.start()
                        self.start_button.setEnabled(True)
                        logger.info(f"Started capturing process: {selected_proc['name']} (PID: {selected_proc['pid']})")
                        QMessageBox.information(
                            self, 
                            "成功", 
                            f"开始捕获进程: {selected_proc['name']} (PID: {selected_proc['pid']})"
                        )
                    else:
                        QMessageBox.warning(self, "错误", "无法捕获选定的进程窗口")
                else:
                    QMessageBox.warning(self, "错误", "未选择有效的进程")
                
        except Exception as e:
            logger.error(f"启动进程捕获时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"启动进程捕获时出错: {str(e)}")

    def on_window_frame_captured(self, frame):
        """处理捕获的窗口画面"""
        try:
            if frame is not None and frame.size > 0:
                # 确保帧数据有效
                if len(frame.shape) == 3 and frame.shape[2] >= 3:
                    self.frame_buffer.append(frame.copy())  # 创建副本避免数据竞争
                    if self.is_inferencing and self.inference_thread:
                        self.inference_thread.update_frame(frame.copy())
                    else:
                        self.display_frame(frame)
                    logger.debug(f"Processed frame shape: {frame.shape}")
                else:
                    logger.warning(f"Invalid frame shape in on_window_frame_captured: {frame.shape}")
            else:
                logger.warning("Received None or empty frame in on_window_frame_captured")
        except Exception as e:
            logger.error(f"Error in on_window_frame_captured: {str(e)}")

    def display_frame(self, frame):
        """显示帧画面"""
        if frame is not None and frame.size > 0:
            try:
                # 确保帧数据有效
                if len(frame.shape) == 3 and frame.shape[2] >= 3:
                    height, width, channel = frame.shape
                    bytes_per_line = 3 * width
                    
                    # 确保使用RGB格式
                    if channel == 4:  # 如果是RGBA格式
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    
                    # 创建QImage
                    q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    
                    # 获取标签大小并保持纵横比
                    label_size = self.image_label.size()
                    scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    
                    # 计算缩放比例
                    self.image_scale = scaled_pixmap.width() / width
                    
                    # 创建新的pixmap用于绘制
                    result_pixmap = QPixmap(label_size)
                    result_pixmap.fill(Qt.transparent)
                    
                    # 创建painter
                    painter = QPainter(result_pixmap)
                    
                    # 计算居中位置
                    x = (label_size.width() - scaled_pixmap.width()) // 2
                    y = (label_size.height() - scaled_pixmap.height()) // 2
                    
                    # 绘制图像
                    painter.drawPixmap(x, y, scaled_pixmap)
                    
                    # 如果有计数线，绘制它
                    if self.counting_line and len(self.counting_line) == 4:
                        painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.SolidLine))
                        scaled_line = [int(p * self.image_scale) for p in self.counting_line]
                        painter.drawLine(scaled_line[0], scaled_line[1], scaled_line[2], scaled_line[3])
                    
                    # 如果显示统计数据
                    if self.show_statistics:
                        painter.setPen(QPen(QColor(0, 255, 0)))
                        painter.setFont(QFont("Arial", 12))
                        
                        # 绘制计数数据
                        y_offset = 30
                        painter.drawText(10, y_offset, f"上行: {self.count_up}")
                        y_offset += 30
                        painter.drawText(10, y_offset, f"下行: {self.count_down}")
                        
                        # 绘制类别计数
                        for class_name, count in self.class_counts.items():
                            y_offset += 30
                            max_count = self.max_class_counts.get(class_name, 0)
                            painter.drawText(10, y_offset, f"{class_name}: {count}/{max_count}")
                        
                        # 绘制经过计数线的次数
                        for class_name, count in self.class_line_counts.items():
                            y_offset += 30
                            painter.drawText(10, y_offset, f"{class_name} 经过: {count}")
                    
                    painter.end()
                    
                    # 显示结果
                    self.image_label.setPixmap(result_pixmap)
                    
                    logger.debug(f"Displayed frame - Original size: {width}x{height}, Scaled size: {scaled_pixmap.width()}x{scaled_pixmap.height()}")
                else:
                    logger.warning(f"Invalid frame shape in display_frame: {frame.shape}")
            except Exception as e:
                logger.error(f"Error in display_frame: {str(e)}")
        else:
            logger.warning("Attempted to display None or empty frame")

    def stop(self):
        """停止所有捕获和推理"""
        logger.info("Stopping all capture and inference...")
        
        # 停止推理
        self.is_inferencing = False
        
        # 停止所有定时器
        self.timer.stop()
        self.capture_timer.stop()
        self.display_timer.stop()
        
        # 停止推理线程
        if self.inference_thread:
            try:
                self.inference_thread.stop()
                self.inference_thread.wait()
                self.inference_thread = None
            except Exception as e:
                logger.error(f"Error stopping inference thread: {str(e)}")
        
        # 停止窗口捕获线程
        if self.window_capture_thread:
            try:
                self.window_capture_thread.stop()
                self.window_capture_thread.wait()
                self.window_capture_thread = None
            except Exception as e:
                logger.error(f"Error stopping window capture thread: {str(e)}")
        
        # 停止摄像头捕获
        if self.cap is not None:
            try:
                self.cap.release()
                self.cap = None
            except Exception as e:
                logger.error(f"Error releasing camera: {str(e)}")
        
        # 关闭屏幕捕获窗口
        if self.capture_window:
            try:
                self.capture_window.hide()
                self.capture_window = None
            except Exception as e:
                logger.error(f"Error hiding capture window: {str(e)}")
        
        # 清除捕获相关的状态
        self.capture_rect = None
        
        # 重置按钮状态
        self.start_button.setText("开始推理")
        self.start_button.setEnabled(True)
        
        logger.info("All capture and inference stopped successfully")

    def clear_counting_line(self):
        """清除计数线"""
        try:
            if self.counting_line is not None:
                self.counting_line = None
                # 重新显示当前帧，但不显示计数线
                if self.last_annotated_frame is not None:
                    self.display_frame(self.last_annotated_frame)
                elif self.frame_buffer:
                    self.display_frame(self.frame_buffer[-1])
                QMessageBox.information(self, "成功", "计数线已清除")
                logger.info("Counting line cleared")
            else:
                QMessageBox.information(self, "提示", "当前没有设置计数线")
        except Exception as e:
            logger.error(f"Error clearing counting line: {str(e)}")
            QMessageBox.critical(self, "错误", f"清除计数线时出错：{str(e)}")

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = ObjectDetectionApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {str(e)}")
        print(f"发生严重错误，请查看日志文件获取详细信息：{str(e)}")