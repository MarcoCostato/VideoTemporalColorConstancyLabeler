# main.py

import sys
import os
import cv2
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QScrollArea, QGridLayout, QStackedWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from imageViewer import FullImageWindow

# -----------------------
# CONFIG
# -----------------------
THUMBNAIL_SIZE = (150, 150)
GRID_COLUMNS = 6
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg')


# -----------------------
# HELPERS
# -----------------------
def load_image_as_qpixmap(path, size=None):
    """Load an image from path, resize, and return QPixmap."""
    img = cv2.imread(path)
    if img is None:
        return QPixmap()
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# -----------------------
# WIDGETS
# -----------------------
class FileTile(QWidget):
    clicked = pyqtSignal(str)

    def __init__(self, folder_name, image_path, frames_path):
        super().__init__()
        self.frames_path = frames_path

        layout = QVBoxLayout(self)
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setPixmap(load_image_as_qpixmap(image_path, THUMBNAIL_SIZE))

        label = QLabel(folder_name)
        label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.img_label)
        layout.addWidget(label)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.frames_path)


class ClickableImage(QLabel):
    clicked = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_path)


# -----------------------
# MAIN APP
# -----------------------
class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Browser")
        self.resize(1000, 800)

        self.delete_mode = False
        self.current_frames_path = None
        self.video_folder = None

        self.layout = QVBoxLayout(self)
        self.btn_load_folder = QPushButton("Select Root Folder")
        self.btn_load_folder.clicked.connect(self.load_folder)

        self.btn_delete_mode = QPushButton("🗑 Delete Mode (Off)")
        self.btn_delete_mode.setCheckable(True)
        self.btn_delete_mode.clicked.connect(self.toggle_delete_mode)

        self.stack = QStackedWidget()
        self.layout.addWidget(self.btn_load_folder)
        self.layout.addWidget(self.stack)

        # Root view
        self.root_layout = QGridLayout()
        root_view = QWidget()
        root_view.setLayout(self.root_layout)
        root_scroll = QScrollArea()
        root_scroll.setWidgetResizable(True)
        root_scroll.setWidget(root_view)
        self.stack.addWidget(root_scroll)

        # Frames view
        self.frames_layout = QGridLayout()
        frames_grid_container = QWidget()
        frames_grid_container.setLayout(self.frames_layout)

        self.btn_back = QPushButton("← Back to Folder View")
        self.btn_back.clicked.connect(self.back_to_root)

        frames_main_layout = QVBoxLayout()
        frames_main_layout.addWidget(self.btn_back)
        frames_main_layout.addWidget(self.btn_delete_mode)

        frames_scroll = QScrollArea()
        frames_scroll.setWidgetResizable(True)
        frames_scroll.setWidget(frames_grid_container)
        frames_main_layout.addWidget(frames_scroll)

        frames_view = QWidget()
        frames_view.setLayout(frames_main_layout)
        self.stack.addWidget(frames_view)

    # -----------------------
    # FOLDER LOGIC
    # -----------------------
    def toggle_delete_mode(self):
        self.delete_mode = self.btn_delete_mode.isChecked()
        self.btn_delete_mode.setText(
            "🗑 Delete Mode (On)" if self.delete_mode else "🗑 Delete Mode (Off)"
        )

    def load_folder(self):
        root_folder = QFileDialog.getExistingDirectory(self, "Select Root Folder")
        if not root_folder:
            return

        # Clear grid
        while self.root_layout.count():
            child = self.root_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        row = col = 0
        for entry in sorted(os.listdir(root_folder)):
            folder_path = os.path.join(root_folder, entry)
            frames_path = os.path.join(folder_path, "processed_labeled_RGB_png", "frames")

            if not os.path.isdir(frames_path):
                continue

            images = [f for f in os.listdir(frames_path) if f.lower().endswith(IMG_EXTENSIONS)]
            if not images:
                continue

            preview_path = os.path.join(frames_path, sorted(images)[0])
            tile = FileTile(entry, preview_path, frames_path)
            tile.clicked.connect(self.open_frames_folder)
            self.root_layout.addWidget(tile, row, col)

            col += 1
            if col >= GRID_COLUMNS:
                col, row = 0, row + 1

        self.stack.setCurrentIndex(0)

    def open_frames_folder(self, frames_path):
        self.current_frames_path = frames_path
        self.video_folder = os.path.dirname(os.path.dirname(frames_path))

        while self.frames_layout.count():
            child = self.frames_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        row = col = 0
        for img_file in sorted(os.listdir(frames_path)):
            if not img_file.lower().endswith(IMG_EXTENSIONS):
                continue

            full_path = os.path.join(frames_path, img_file)
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(5, 5, 5, 5)
            vbox.setSpacing(2)

            thumb = ClickableImage(full_path)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setPixmap(load_image_as_qpixmap(full_path, THUMBNAIL_SIZE))
            thumb.clicked.connect(lambda path=full_path: self.image_clicked(path))

            name_label = QLabel(img_file)
            name_label.setAlignment(Qt.AlignCenter)

            vbox.addWidget(thumb)
            vbox.addWidget(name_label)
            self.frames_layout.addWidget(container, row, col)

            col += 1
            if col >= GRID_COLUMNS:
                col, row = 0, row + 1

        self.stack.setCurrentIndex(1)

    # -----------------------
    # IMAGE LOGIC
    # -----------------------
    def image_clicked(self, image_path):
        if self.delete_mode:
            reply = QMessageBox.question(
                self, "Confirm Delete",
                f"Delete {os.path.basename(image_path)} and related files?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.delete_image_and_data(image_path)
                self.open_frames_folder(self.current_frames_path)
        else:
            self.open_full_image(image_path)

    def delete_image_and_data(self, image_path):
        img_name = os.path.basename(image_path)
        img_stem = os.path.splitext(img_name)[0]

        if os.path.exists(image_path):
            os.remove(image_path)

        csvs_path = os.path.join(self.video_folder, "processed_labeled_RGB_png", "CSVs")
        csv_file = os.path.join(csvs_path, f"{img_stem}.csv")
        if os.path.exists(csv_file):
            os.remove(csv_file)

        labels_csv_path = os.path.join(self.video_folder, "labels.csv")
        if os.path.exists(labels_csv_path):
            df = pd.read_csv(labels_csv_path)
            df = df[df["frame"] != img_name]
            df.to_csv(labels_csv_path, index=False)

    def back_to_root(self):
        self.stack.setCurrentIndex(0)

    def open_full_image(self, image_path):
        viewer = FullImageWindow(image_path)
        viewer.exec_()

    def changeEvent(self, event):
        if event.type() == QEvent.ActivationChange:
            if self.isActiveWindow() and self.stack.currentIndex() == 1:
                if self.current_frames_path:
                    self.open_frames_folder(self.current_frames_path)
        super().changeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

