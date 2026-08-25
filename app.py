import sys
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QSize, QPointF
from PySide6.QtGui import QImage, QPixmap, QPainter, QIcon, QPen, QBrush, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QFrame,
    QListWidget, QListWidgetItem, QSizePolicy
)


APP_NAME = "Scan Pro AI"
MODELS_DIR = Path(__file__).resolve().parent / "models"
EDSR_URL = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb"
EDSR_FILE = MODELS_DIR / "EDSR_x2.pb"


def desktop_path():
    path = Path.home() / "Desktop"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def order_points(points):
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def detect_document_corners(image):
    h, w = image.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    image_area = small.shape[0] * small.shape[1]
    candidates = []

    edges = cv2.Canny(gray, 35, 130)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), 2
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(contours)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), 2
    )
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(contours)

    best = None
    best_score = -1.0
    for c in candidates:
        area = cv2.contourArea(c)
        if area < image_area * 0.18:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.025 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        rect_area = abs(cv2.contourArea(pts))
        if rect_area <= 0:
            continue
        rectangularity = min(1.0, area / rect_area)
        score = (area / image_area) * 2.5 + rectangularity
        if score > best_score:
            best_score = score
            best = pts

    if best is not None:
        if scale < 1:
            best /= scale
        return order_points(best)

    mx, my = w * 0.04, h * 0.04
    return np.array([
        [mx, my], [w - mx, my],
        [w - mx, h - my], [mx, h - my]
    ], dtype=np.float32)


def perspective_transform(image, corners):
    pts = order_points(corners)
    tl, tr, br, bl = pts
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width = max(300, min(width, 6000))
    height = max(300, min(height, 8000))

    dst = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, matrix, (width, height),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def document_ai_enhance(img):
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    norm = cv2.divide(gray, morph, scale=255.0)
    norm = np.uint8(norm)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)

    _, final_gray = cv2.threshold(enhanced, 215, 255, cv2.THRESH_TRUNC)
    final_gray = cv2.normalize(final_gray, None, 0, 255, norm_type=cv2.NORM_MINMAX)

    h, w = final_gray.shape
    margin_x = max(1, int(w * 0.015))
    margin_y = max(1, int(h * 0.015))
    
    final_gray[:margin_y, :] = 255
    final_gray[-margin_y:, :] = 255
    final_gray[:, :margin_x] = 255
    final_gray[:, -margin_x:] = 255

    return cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)


def ai_super_resolution(img):
    try:
        if not hasattr(cv2, "dnn_superres"):
            return None
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if not EDSR_FILE.exists():
            try:
                urllib.request.urlretrieve(EDSR_URL, str(EDSR_FILE))
            except Exception:
                return None
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(EDSR_FILE))
        sr.setModel("edsr", 2)
        h, w = img.shape[:2]
        max_side = 2200
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            work = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            work = img
        return sr.upsample(work)
    except Exception:
        return None


def magic_pro_ai(image, corners):
    scanned = perspective_transform(image, corners)
    ai = ai_super_resolution(scanned)
    if ai is not None:
        scanned = ai
    return document_ai_enhance(scanned)


def svg_icon(kind, color="#111111"):
    icons = {
        "rotate_single": f'<path d="M24 8C15.16 8 8 15.16 8 24s7.16 16 16 16c7.05 0 13-4.56 15.1-10.8" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round"/><path d="M39 12v12H27" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>',
        "original": f'<rect x="10" y="6" width="28" height="36" rx="3" fill="{color}" opacity=".12"/><rect x="10" y="6" width="28" height="36" rx="3" fill="none" stroke="{color}" stroke-width="2.5"/><path d="M16 16h16M16 23h16M16 30h12M16 37h8" stroke="{color}" stroke-width="2.5"/>',
        "ai": '<text x="4" y="36" font-family="Arial" font-size="30" font-weight="700" fill="#00B89C">AI</text><path d="M39 7l2 6 6 2-6 2-2 6-2-6-6-2 6-2z" fill="#18C9A7"/>',
        "plus": '<path d="M24 10v28M10 24h28" stroke="#999999" stroke-width="4" stroke-linecap="round"/>',
        "trash": f'<path d="M12 14h24M18 14V10h12v4M15 14v22a2 2 0 002 2h14a2 2 0 002-2V14" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><path d="M20 20v10M28 20v10" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
    }
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">{icons[kind]}</svg>'


def make_icon(kind, color="#111111", size=46):
    from PySide6.QtSvg import QSvgRenderer
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    QSvgRenderer(svg_icon(kind, color).encode("utf-8")).render(painter)
    painter.end()
    return QIcon(pix)


class InteractivePreview(QLabel):
    """Preview widget with interactive 4 corner handles."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#FFFFFF; color:#222; border-radius:4px;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image = None
        self.base_image = None
        self.corners = None
        self.active_handle = -1
        self.corner_changed_callback = None

    def set_data(self, image, corners=None, base_image=None):
        self.image = image
        if base_image is not None:
            self.base_image = base_image
        elif self.base_image is None:
            self.base_image = image

        if corners is not None:
            self.corners = corners.copy()
        self.update()

    def get_image_rect(self):
        img_to_check = self.base_image if self.base_image is not None else self.image
        if img_to_check is None:
            return None
        h, w = img_to_check.shape[:2]
        pw, ph = self.width(), self.height()
        scale = min(pw / w, ph / h)
        nw, nh = int(w * scale), int(h * scale)
        ox = (pw - nw) // 2
        oy = (ph - nh) // 2
        return ox, oy, nw, nh, scale

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image is None:
            return

        rect_info = self.get_image_rect()
        if not rect_info:
            return
        ox, oy, nw, nh, scale = rect_info

        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        
        # Calculate scaled dimensions for current displayed image
        ih, iw = self.image.shape[:2]
        iscale = min(self.width() / iw, self.height() / ih)
        inw, inh = int(iw * iscale), int(ih * iscale)
        iox = (self.width() - inw) // 2
        ioy = (self.height() - inh) // 2

        pix = QPixmap.fromImage(qimg).scaled(inw, inh, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        painter = QPainter(self)
        painter.drawPixmap(iox, ioy, pix)

        # Always draw interactive 4 corner handles over the view
        if self.corners is not None:
            pts_disp = []
            for pt in self.corners:
                x = ox + pt[0] * scale
                y = oy + pt[1] * scale
                pts_disp.append(QPointF(x, y))

            pen = QPen(QColor("#10B99A"), 2, Qt.DashLine)
            painter.setPen(pen)
            for i in range(4):
                painter.drawLine(pts_disp[i], pts_disp[(i + 1) % 4])

            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.setBrush(QBrush(QColor("#10B99A")))
            for p in pts_disp:
                painter.drawEllipse(p, 8, 8)

        painter.end()

    def mousePressEvent(self, event):
        if self.corners is None:
            return
        rect_info = self.get_image_rect()
        if not rect_info:
            return
        ox, oy, _, _, scale = rect_info

        pos = event.position()
        for i, pt in enumerate(self.corners):
            cx = ox + pt[0] * scale
            cy = oy + pt[1] * scale
            if (pos.x() - cx) ** 2 + (pos.y() - cy) ** 2 <= 400:  # 20px hit radius
                self.active_handle = i
                break

    def mouseMoveEvent(self, event):
        if self.active_handle != -1 and self.corners is not None and self.base_image is not None:
            rect_info = self.get_image_rect()
            if not rect_info:
                return
            ox, oy, _, _, scale = rect_info
            pos = event.position()

            h, w = self.base_image.shape[:2]
            x = max(0, min((pos.x() - ox) / scale, w))
            y = max(0, min((pos.y() - oy) / scale, h))

            self.corners[self.active_handle] = [x, y]
            self.update()

    def mouseReleaseEvent(self, event):
        if self.active_handle != -1:
            self.active_handle = -1
            if self.corner_changed_callback:
                self.corner_changed_callback(self.corners)


class ScanPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1450, 850)
        self.setMinimumSize(1000, 650)

        self.original = None
        self.magic = None
        self.corners = None
        self.build_ui()

    def build_ui(self):
        self.setStyleSheet("""
        QMainWindow, QWidget { background:#FFFFFF; color:#202020; font-family:'Segoe UI'; }
        #left { border-right:1px solid #E4E5E7; background:#FFFFFF; }
        #center { background:#F1F2F4; }
        #right { border-left:1px solid #E4E5E7; background:#FFFFFF; }
        #topline { border-bottom:1px solid #E5E6E8; }
        QPushButton { border:none; }
        QPushButton#rotate { background:#1677FF; border-radius:10px; min-width:80px; min-height:80px; }
        QPushButton#rotate:hover { background:#0D5BCC; }
        QPushButton#tool { background:#F0F0F0; border-radius:9px; min-width:92px; min-height:80px; }
        QPushButton#tool:hover { background:#E8E8E8; }
        QPushButton#tool[selected='true'] { background:#D9F7F1; border:1px solid #10B99A; }
        QPushButton#delete { background:#F0F0F0; border-radius:9px; min-width:92px; min-height:80px; }
        QPushButton#delete:hover { background:#FFE6E6; }
        QLabel#toolText { font-size:14px; }
        QPushButton#save { background:#10B99A; color:white; border-radius:8px; font-size:18px; font-weight:bold; min-width:160px; min-height:45px; padding:0 20px; }
        QPushButton#save:hover { background:#0DAE91; }
        QPushButton#add { background:#F0F0F0; border-radius:12px; }
        QPushButton#add:hover { background:#E2E2E2; }
        QListWidget { border:none; background:#FFFFFF; }
        QListWidget::item:selected { background:#D9F7F1; border-radius:8px; }
        """)

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        top = QFrame()
        top.setObjectName("topline")
        top.setFixedHeight(12)
        main.addWidget(top)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Left list
        left = QFrame()
        left.setObjectName("left")
        left.setFixedWidth(145)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 20, 10, 15)
        self.pages = QListWidget()
        self.pages.setIconSize(QSize(105, 135))
        self.pages.currentRowChanged.connect(self.select_page)
        ll.addWidget(self.pages)
        body.addWidget(left)

        # Center preview
        center = QFrame()
        center.setObjectName("center")
        cl = QVBoxLayout(center)
        cl.setContentsMargins(24, 24, 24, 24)

        self.preview = InteractivePreview()
        self.preview.corner_changed_callback = self.on_corners_updated
        cl.addWidget(self.preview, 1)

        self.add_button = QPushButton()
        self.add_button.setObjectName("add")
        self.add_button.setIcon(make_icon("plus", size=64))
        self.add_button.setIconSize(QSize(64, 64))
        self.add_button.setFixedSize(120, 120)
        self.add_button.clicked.connect(self.open_image)

        self.add_overlay = QFrame(self.preview)
        ol = QVBoxLayout(self.add_overlay)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(self.add_button, 0, Qt.AlignCenter)
        self.add_overlay.setStyleSheet("background:transparent;")
        self.add_overlay.raise_()
        body.addWidget(center, 1)

        # Right toolbar
        right = QFrame()
        right.setObjectName("right")
        right.setFixedWidth(270)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(28, 40, 28, 20)
        rv.setSpacing(25)

        self.rotate_btn = QPushButton()
        self.rotate_btn.setObjectName("rotate")
        self.rotate_btn.setFixedSize(85, 85)
        self.rotate_btn.setIcon(make_icon("rotate_single", color="#FFFFFF", size=50))
        self.rotate_btn.setIconSize(QSize(50, 50))
        self.rotate_btn.clicked.connect(lambda: self.rotate(1))
        rv.addWidget(self.rotate_btn, 0, Qt.AlignHCenter)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#E5E5E5;")
        rv.addWidget(line)

        self.original_btn = self.tool_button("original", "Original", "#1677FF", self.restore_original)
        self.magic_btn = self.tool_button("ai", "Magic Pro AI", "#00B89C", self.run_magic)
        self.delete_btn = self.tool_button("trash", "", "#D32F2F", self.delete_image, is_delete=True)

        self.original_btn.button.setProperty("selected", False)
        self.magic_btn.button.setProperty("selected", True)

        rv.addWidget(self.original_btn, 0, Qt.AlignHCenter)
        rv.addWidget(self.magic_btn, 0, Qt.AlignHCenter)
        rv.addWidget(self.delete_btn, 0, Qt.AlignHCenter)
        rv.addStretch(1)

        body.addWidget(right)
        main.addLayout(body, 1)

        bottom = QFrame()
        bottom.setFixedHeight(75)
        bottom.setStyleSheet("border-top:1px solid #E5E6E8; background:#FFFFFF;")
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(25, 10, 25, 10)
        bl.addStretch(1)
        self.save_btn = QPushButton("SAVE")
        self.save_btn.setObjectName("save")
        self.save_btn.clicked.connect(self.save_image)
        bl.addWidget(self.save_btn)
        main.addWidget(bottom)

        self.update_overlay()

    def tool_button(self, kind, text, color, slot, is_delete=False):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        b = QPushButton()
        b.setObjectName("delete" if is_delete else "tool")
        b.setIcon(make_icon(kind, color, 52))
        b.setIconSize(QSize(52, 52))
        b.clicked.connect(slot)
        lay.addWidget(b)
        if text:
            label = QLabel(text)
            label.setObjectName("toolText")
            label.setAlignment(Qt.AlignCenter)
            lay.addWidget(label)
        box.button = b
        return box

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_overlay()

    def update_overlay(self):
        if not hasattr(self, "add_overlay"):
            return
        if self.original is None:
            w, h = 130, 130
            x = max(0, (self.preview.width() - w) // 2)
            y = max(0, (self.preview.height() - h) // 2)
            self.add_overlay.setGeometry(x, y, w, h)
            self.add_overlay.show()
            self.add_overlay.raise_()
        else:
            self.add_overlay.hide()

    def open_image(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Import Images", desktop_path(),
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if not fn:
            return

        image = cv2.imread(fn, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.critical(self, "Error", "Cannot open this image.")
            return

        self.original = image
        self.corners = detect_document_corners(image)
        self.magic = None

        h, w = image.shape[:2]
        scale = min(105 / w, 135 / h)
        pw, ph = int(w * scale), int(h * scale)
        small = cv2.resize(image, (pw, ph), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, pw, ph, rgb.strides[0], QImage.Format_RGB888)

        item = QListWidgetItem()
        item.setIcon(QIcon(QPixmap.fromImage(qimg)))
        item.setSizeHint(QSize(115, 150))
        self.pages.clear()
        self.pages.addItem(item)
        self.pages.setCurrentRow(0)

        self.run_magic()
        self.update_overlay()

    def on_corners_updated(self, updated_corners):
        self.corners = updated_corners
        if self.magic_btn.button.property("selected"):
            self.run_magic()

    def select_page(self, row):
        if row < 0 or self.original is None:
            return
        self.restore_original()

    def restore_original(self):
        if self.original is None:
            return
        self.preview.set_data(self.original, self.corners, base_image=self.original)
        self.original_btn.button.setProperty("selected", True)
        self.magic_btn.button.setProperty("selected", False)
        self.update_button_styles()

    def run_magic(self):
        if self.original is None:
            self.open_image()
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            self.magic = magic_pro_ai(self.original, self.corners)
            self.preview.set_data(self.magic, self.corners, base_image=self.original)
            self.magic_btn.button.setProperty("selected", True)
            self.original_btn.button.setProperty("selected", False)
            self.update_button_styles()
        except Exception as exc:
            QMessageBox.warning(self, "Magic Pro AI", f"AI processing failed:\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def update_button_styles(self):
        for btn in (self.original_btn.button, self.magic_btn.button):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def rotate(self, direction):
        if self.original is None:
            return
        self.original = cv2.rotate(self.original, cv2.ROTATE_90_CLOCKWISE)
        self.corners = detect_document_corners(self.original)
        if self.magic_btn.button.property("selected"):
            self.run_magic()
        else:
            self.restore_original()

    def delete_image(self):
        self.original = None
        self.magic = None
        self.corners = None
        self.pages.clear()
        self.preview.set_data(None, None, None)
        self.update_overlay()

    def save_image(self):
        if self.original is None:
            QMessageBox.information(self, "Save", "Import an image first.")
            return
        image = self.magic if self.magic is not None else self.original
        default = os.path.join(desktop_path(), "ScanPro.jpg")
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Image", default,
            "JPEG Image (*.jpg *.jpeg);;PNG Image (*.png)"
        )
        if not fn:
            return
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            fn += ".jpg"
        ok = cv2.imwrite(fn, image, [cv2.IMWRITE_JPEG_QUALITY, 98] if fn.lower().endswith((".jpg", ".jpeg")) else [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            QMessageBox.critical(self, "Save", "Could not save the image.")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = ScanPro()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
