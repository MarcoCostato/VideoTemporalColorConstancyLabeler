import os
import ast
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QComboBox, QPushButton, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QEvent


# ------------------ Helper Classes ------------------

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


# ------------------ Main FullImageWindow ------------------

class FullImageWindow(QDialog):
    def __init__(self, processed_image_path):
        super().__init__()
        self.setWindowTitle("Full Image Viewer")
        self.resize(1000, 800)

        # State variables
        self.processed_image_path = processed_image_path
        self.image_path = self._get_rgb_image_path()
        self.coords_list = []
        self.white_triplets = []
        self.loaded_pixmap = None
        self.original_image = None
        self.processed_image_array = None
        self.pick_mode = False
        self.selected_point_index = 0

        # Layout
        layout = QVBoxLayout(self)

        self.btn_fix_detection = QPushButton("Fix Detection")
        self.btn_fix_detection.clicked.connect(self.load_and_show_coords)
        layout.addWidget(self.btn_fix_detection)

        self.btn_select_coord = QPushButton("Select New Coordinate")
        self.btn_select_coord.clicked.connect(self.start_pick_mode)
        layout.addWidget(self.btn_select_coord)

        self.coord_selector = QComboBox()
        self.coord_selector.currentIndexChanged.connect(self.update_dot_display)
        self.coord_selector.setVisible(False)
        layout.addWidget(self.coord_selector)

        # Graphics view
        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.viewport().installEventFilter(self)
        layout.addWidget(self.view)

        # Load initial image
        self._load_image()

    # ---------- File path helpers ----------

    def _get_rgb_image_path(self):
        """Get the original RGB image path from processed path."""
        filename = os.path.basename(self.processed_image_path)
        video_folder = os.path.dirname(os.path.dirname(self.processed_image_path))
        root_folder = os.path.dirname(video_folder)
        rgb_image_path = os.path.join(root_folder, "RGB_png", filename)

        if not os.path.exists(rgb_image_path):
            QMessageBox.warning(self, "Missing Image", f"RGB image not found:\n{rgb_image_path}")
            return self.processed_image_path
        return rgb_image_path

    def _get_csv_path(self):
        img_name = os.path.basename(self.image_path)
        parent_dir = os.path.dirname(os.path.dirname(self.image_path))
        return os.path.join(parent_dir, "processed_labeled_RGB_png", "CSVs", img_name.replace(".png", ".csv"))

    def _get_labels_csv_path(self):
        parent_dir = os.path.dirname(os.path.dirname(self.image_path))
        return os.path.join(parent_dir, "labels.csv")

    # ---------- Image loading & processing ----------

    def _load_image(self):
        """Load and process the image using WB parameters from CSV."""
        rgb = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if rgb is None:
            QMessageBox.critical(self, "Error", f"Failed to read image:\n{self.image_path}")
            return

        self.original_image = rgb.copy()
        rgb = rgb.astype(np.float32) / 255.0

        csv_file_path = self._get_csv_path()
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
            self.white_triplets = []
            for val in df.get("White", []):
                try:
                    triplet = np.array([float(x) / 255 for x in val.strip("[]").split(",")], dtype=np.float32)
                    if triplet.shape == (3,):
                        self.white_triplets.append(triplet)
                except Exception:
                    pass

        if not self.white_triplets:
            QMessageBox.warning(self, "Missing Data", "No valid white triplets found.")
            return

        wb_params = np.mean(self.white_triplets, axis=0)
        wb_params /= np.linalg.norm(wb_params)
        rgb = np.clip(rgb / (wb_params + 1e-6), 0.0, 1.0)
        rgb = (rgb * 255).astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        self.processed_image_array = rgb.copy()
        self._update_scene_from_array(rgb)

    def _update_scene_from_array(self, rgb_array):
        """Display a numpy RGB image in the QGraphicsScene."""
        h, w, ch = rgb_array.shape
        qimg = QImage(rgb_array.data, w, h, ch * w, QImage.Format_RGB888)
        self.loaded_pixmap = QPixmap.fromImage(qimg)
        self.scene.clear()
        self.image_item = QGraphicsPixmapItem(self.loaded_pixmap)
        self.scene.addItem(self.image_item)

    # ---------- Event handling ----------

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton and self.pick_mode:
            pos = self.view.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            self._handle_image_click(x, y)
            return True
        return super().eventFilter(source, event)

    # ---------- UI actions ----------

    def load_and_show_coords(self):
        """Load coords from CSV and show first dot."""
        csv_file_path = self._get_csv_path()
        if not os.path.exists(csv_file_path):
            QMessageBox.warning(self, "CSV Not Found", f"No CSV found for {os.path.basename(self.image_path)}")
            return

        df = pd.read_csv(csv_file_path)
        if 'White_coords' not in df.columns:
            QMessageBox.warning(self, "Invalid CSV", f"No 'White_coords' column in {csv_file_path}")
            return

        self.coords_list = [ast.literal_eval(str(coord)) for coord in df['White_coords'].dropna()]
        if not self.coords_list:
            QMessageBox.information(self, "No Coordinates", "No valid white coordinates found.")
            return

        self.coord_selector.clear()
        for i, coord in enumerate(self.coords_list):
            self.coord_selector.addItem(f"Point {i+1}: {coord}")

        self.coord_selector.setVisible(True)
        self.update_dot_display(0)

    def update_dot_display(self, index):
        """Draw selected dot on image."""
        if not self.loaded_pixmap or not self.coords_list:
            return
        pixmap_copy = self.loaded_pixmap.copy()
        painter = QPainter(pixmap_copy)
        pen = QPen(Qt.red)
        pen.setWidth(4)
        painter.setPen(pen)
        try:
            x, y = self.coords_list[index]
            painter.drawPoint(int(x), int(y))
        except Exception:
            pass
        painter.end()
        self.image_item.setPixmap(pixmap_copy)

    def start_pick_mode(self):
        if not self.coords_list:
            QMessageBox.warning(self, "No Points", "Load detection first.")
            return
        self.selected_point_index = self.coord_selector.currentIndex()
        self.pick_mode = True
        QMessageBox.information(self, "Pick Mode", "Click on the image to select a new coordinate.")

    def _handle_image_click(self, x, y):
        """Handle new coordinate selection and update CSVs."""
        # Bounds check
        if (x < 0 or y < 0 or x >= self.original_image.shape[1] or y >= self.original_image.shape[0]):
            QMessageBox.warning(self, "Invalid Click", "Clicked outside image bounds.")
            return

        clicked_rgb = self.original_image[y, x].astype(np.float32) / 255.0
        self.coords_list[self.selected_point_index] = [x, y]
        self.white_triplets[self.selected_point_index] = clicked_rgb

        # Update CSV in CSVs folder
        csv_path = self._get_csv_path()
        df = pd.read_csv(csv_path)
        if self.selected_point_index < len(df):
            df.loc[self.selected_point_index, "White_coords"] = f"[{x}, {y}]"
            df.loc[self.selected_point_index, "White"] = f"[{int(clicked_rgb[0]*255)}, {int(clicked_rgb[1]*255)}, {int(clicked_rgb[2]*255)}]"
        df.to_csv(csv_path, index=False)

        # Update labels.csv
        labels_path = self._get_labels_csv_path()
        labels_df = pd.read_csv(labels_path)
        mean_triplet = np.mean(self.white_triplets, axis=0)
        labels_df.loc[labels_df["frame"] == os.path.basename(self.image_path), ["red", "green", "blue"]] = [int(mean_triplet[0]*255), int(mean_triplet[1]*255), int(mean_triplet[2]*255)]
        labels_df.to_csv(labels_path, index=False)

        # Reset pick mode and refresh
        self.pick_mode = False
        self._load_image()
        self._save_processed_image()
        QMessageBox.information(self, "Updated", "Coordinate and CSV values updated.")

    def _save_processed_image(self):
        img_name = os.path.basename(self.image_path)
        parent_dir = os.path.dirname(os.path.dirname(self.image_path))
        save_path = os.path.join(parent_dir, "processed_labeled_RGB_png", "frames", img_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        bgr = cv2.cvtColor(self.processed_image_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr)
