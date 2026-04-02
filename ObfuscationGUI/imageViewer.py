import os
import cv2
import pandas as pd
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QPen, QColor, QBrush, QPainter
from PyQt5.QtCore import Qt, QRectF


class ZoomableGraphicsView(QGraphicsView):
    """QGraphicsView that supports mouse wheel zooming and drag panning."""
    def __init__(self, parent=None, max_zoom=10):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 0
        self.max_zoom = max_zoom

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:  # zoom in
            if self._zoom < self.max_zoom:
                self.scale(zoom_in_factor, zoom_in_factor)
                self._zoom += 1
        else:  # zoom out
            if self._zoom > 0:
                self.scale(zoom_out_factor, zoom_out_factor)
                self._zoom -= 1
        if self._zoom == 0:
            self.resetTransform()


class HandleItem(QGraphicsRectItem):
    """Corner/edge handle that *doesn't move itself*; it drives parent resize."""
    SIZE = 10

    def __init__(self, cursor_type, position_flag, parent=None):
        super().__init__(-HandleItem.SIZE/2, -HandleItem.SIZE/2, HandleItem.SIZE, HandleItem.SIZE, parent)
        self.setBrush(QBrush(QColor('red')))
        self.setPen(QPen(Qt.NoPen))
        self.setCursor(cursor_type)
        # Don’t actually move the handle item; we’ll resize parent on mouse move.
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)  # constant screen size
        self.position_flag = position_flag
        self.setZValue(2)  # above the black rect

    def mousePressEvent(self, event):
        event.accept()

    def mouseMoveEvent(self, event):
        # Map mouse to parent (rect item) coordinates and ask parent to resize
        parent = self.parentItem()
        if parent is not None:
            p = parent.mapFromScene(event.scenePos())
            parent.resize_via_handle(self.position_flag, p)
        event.accept()

    def mouseReleaseEvent(self, event):
        event.accept()


class ResizableRectItem(QGraphicsRectItem):
    """Solid black rectangle with 8 resize handles; can also be moved."""
    HANDLE_CURSORS = {
        "top_left": Qt.SizeFDiagCursor,
        "top": Qt.SizeVerCursor,
        "top_right": Qt.SizeBDiagCursor,
        "right": Qt.SizeHorCursor,
        "bottom_right": Qt.SizeFDiagCursor,
        "bottom": Qt.SizeVerCursor,
        "bottom_left": Qt.SizeBDiagCursor,
        "left": Qt.SizeHorCursor,
    }

    def __init__(self, rect):
        super().__init__(rect)
        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        # Solid black, no border
        self.setBrush(QBrush(QColor(0, 0, 0)))
        self.setPen(QPen(Qt.NoPen))
        self.setZValue(1)

        # Create handle items (children)
        self.handles = {}
        for pos_name, cursor in self.HANDLE_CURSORS.items():
            self.handles[pos_name] = HandleItem(cursor, pos_name, self)

        self.update_handles_pos()

    def update_handles_pos(self):
        r = self.rect()
        self.handles["top_left"].setPos(r.topLeft())
        self.handles["top"].setPos(r.left() + r.width()/2, r.top())
        self.handles["top_right"].setPos(r.topRight())
        self.handles["right"].setPos(r.right(), r.top() + r.height()/2)
        self.handles["bottom_right"].setPos(r.bottomRight())
        self.handles["bottom"].setPos(r.left() + r.width()/2, r.bottom())
        self.handles["bottom_left"].setPos(r.bottomLeft())
        self.handles["left"].setPos(r.left(), r.top() + r.height()/2)

    def resize_via_handle(self, handle_flag, p):
        """Resize the rectangle using a given handle to the parent-local point p."""
        rect = self.rect()
        new_rect = QRectF(rect)

        if handle_flag == "top_left":
            new_rect.setTopLeft(p)
        elif handle_flag == "top":
            new_rect.setTop(p.y())
        elif handle_flag == "top_right":
            new_rect.setTopRight(p)
        elif handle_flag == "right":
            new_rect.setRight(p.x())
        elif handle_flag == "bottom_right":
            new_rect.setBottomRight(p)
        elif handle_flag == "bottom":
            new_rect.setBottom(p.y())
        elif handle_flag == "bottom_left":
            new_rect.setBottomLeft(p)
        elif handle_flag == "left":
            new_rect.setLeft(p.x())

        # Enforce minimum size and prevent inverted rects
        min_size = 20
        if new_rect.width() < min_size:
            if handle_flag in ["top_left", "left", "bottom_left"]:
                new_rect.setLeft(new_rect.right() - min_size)
            else:
                new_rect.setRight(new_rect.left() + min_size)
        if new_rect.height() < min_size:
            if handle_flag in ["top_left", "top", "top_right"]:
                new_rect.setTop(new_rect.bottom() - min_size)
            else:
                new_rect.setBottom(new_rect.top() + min_size)
        if new_rect.left() > new_rect.right():
            new_rect.setLeft(new_rect.right())
        if new_rect.top() > new_rect.bottom():
            new_rect.setTop(new_rect.bottom())

        self.prepareGeometryChange()
        self.setRect(new_rect)
        self.update_handles_pos()
        self.update()

    def itemChange(self, change, value):
        # Keep handles glued to corners when the whole rect moves
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.update_handles_pos()
        return super().itemChange(change, value)


class FullImageWindow(QDialog):
    def __init__(self, processed_image_path):
        super().__init__()
        self.setWindowTitle("Full Image Viewer")
        self.resize(1000, 800)
        self.processed_image_path = processed_image_path

        # Main vertical layout
        layout = QVBoxLayout(self)

        # Graphics View + Scene
        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self)
        self.view.setScene(self.scene)
        layout.addWidget(self.view)

        # Buttons row
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("💾 Save Boxes")
        self.btn_save.clicked.connect(self.save_boxes)
        self.btn_add = QPushButton("➕ Add Box")
        self.btn_add.clicked.connect(self.add_box)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_add)
        layout.addLayout(btn_layout)

        # Load the image + boxes
        self._load_image_and_boxes()

    def keyPressEvent(self, event):
        """Delete selected rectangles with Delete/Backspace."""
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for item in self.scene.selectedItems():
                if isinstance(item, ResizableRectItem):
                    self.scene.removeItem(item)
            event.accept()
        else:
            super().keyPressEvent(event)

    def _load_image_and_boxes(self):
        """Load the main image and any saved boxes from CSV."""
        if not os.path.exists(self.processed_image_path):
            print(f"Image not found: {self.processed_image_path}")
            return

        pixmap = QPixmap(self.processed_image_path)
        if pixmap.isNull():
            print(f"Failed to load image: {self.processed_image_path}")
            return

        self.scene.clear()
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)

        csv_path = self._get_boxes_csv_path()
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    y1 = int(row['bbox_y1'])
                    x1 = int(row['bbox_x1'])
                    y2 = int(row['bbox_y2'])
                    x2 = int(row['bbox_x2'])
                    rect_item = ResizableRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
                    self.scene.addItem(rect_item)
            except Exception as e:
                print(f"Error reading CSV {csv_path}: {e}")

    def _get_boxes_csv_path(self):
        """Return the path to the CSV file for this image."""
        filename = os.path.basename(self.processed_image_path).replace(".png", ".csv")
        video_folder = os.path.dirname(os.path.dirname(self.processed_image_path))
        return os.path.join(video_folder, "CSVs", "BoxesCSVs", filename)

    def save_boxes(self):
        """Save box coordinates to CSV and draw them on an image."""
        boxes = []
        for item in self.scene.items():
            if isinstance(item, ResizableRectItem):
                scene_rect = item.mapRectToScene(item.rect())
                x1 = int(scene_rect.left())
                y1 = int(scene_rect.top())
                x2 = int(scene_rect.right())
                y2 = int(scene_rect.bottom())
                boxes.append([y1, x1, y2, x2])

        csv_path = self._get_boxes_csv_path()
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = pd.DataFrame(boxes, columns=['bbox_y1', 'bbox_x1', 'bbox_y2', 'bbox_x2'])
        df.to_csv(csv_path, index=False)
        print(f"Saved updated boxes to {csv_path}")

        # Save image with opaque black boxes
        pixmap = QPixmap(self.processed_image_path)
        painter = QPainter(pixmap)
        painter.setBrush(QColor(0, 0, 0, 255))  # Fully opaque black
        painter.setPen(Qt.NoPen)
        for y1, x1, y2, x2 in boxes:
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
        painter.end()

        boxes_img_path = self.processed_image_path.replace(
            os.path.join("processed_labeled_RGB_png", "frames"),
            os.path.join("processed_labeled_RGB_png", "frames", "boxes")
        )
        os.makedirs(os.path.dirname(boxes_img_path), exist_ok=True)
        pixmap.save(boxes_img_path)
        print(f"Saved image with boxes to {boxes_img_path}")

    def add_box(self):
        """Add a new rectangle in the center of the image."""
        if not hasattr(self, "image_item"):
            return
        rect = self.image_item.boundingRect()
        w, h = 100, 80  # default size
        center_x = rect.width() / 2 - w / 2
        center_y = rect.height() / 2 - h / 2
        rect_item = ResizableRectItem(QRectF(center_x, center_y, w, h))
        self.scene.addItem(rect_item)
