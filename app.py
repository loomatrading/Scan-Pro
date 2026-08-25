import sys
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QIcon
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
    """Fast document detection. Works on a downscaled copy, then maps points back."""
    h, w = image.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    image_area = small.shape[0] * small.shape[1]
    candidates = []

    # Edges are reliable for paper documents.
    edges = cv2.Canny(gray, 35, 130)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), 2
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(contours)

    # Bright page fallback.
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

    # No detected border: use a safe 4% margin instead of failing.
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
    """Scanner-style enhancement: denoise, illumination correction, local contrast and sharpening."""
    if img is None:
        return None

    # Work in LAB for stable document contrast.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Remove camera noise while preserving text edges.
    result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21)

    # Even out shadows/light falloff on paper.
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 25)
    background = np.maximum(background, 1)
    normalized = cv2.divide(gray, background, scale=235)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = normalized
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Gentle unsharp mask for text clarity.
    blur = cv2.GaussianBlur(result, (0, 0), 1.0)
    result = cv2.addWeighted(result, 1.18, blur, -0.18, 0)
    return result


def ai_super_resolution(img):
    """Real neural EDSR x2 when the model is available; graceful fallback otherwise."""
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
        # EDSR is expensive on very large images; process a reasonable working size.
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
    """One-click scanner pipeline: perspective correction + real AI super-resolution + scanner enhancement."""
    scanned = perspective_transform(image, corners)
    ai = ai_super_resolution(scanned)
    if ai is not None:
        scanned = ai
    return document_ai_enhance(scanned)


def cv_to_pixmap(image, max_w=900, max_h=700):
    if image is None:
        return QPixmap()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def svg_icon(kind, color="#111111"):
    icons = {
        "left": f'<path d="M35 10L17 24l18 14" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><path d="M18 24h23" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
        "right": f'<path d="M13 10l18 14-18 14" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><path d="M30 24H7" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
        "original": f'<rect x="10" y="6" width="28" height="36" rx="3" fill="{color}" opacity=".12"/><rect x="10" y="6" width="28" height="36" rx="3" fill="none" stroke="{color}" stroke-width="2.5"/><path d="M16 16h16M16 23h16M16 30h12M16 37h8" stroke="{color}" stroke-width="2.5"/>',
        "ai": '<text x="4" y="36" font-family="Arial" font-size="30" font-weight="700" fill="#00B89C">AI</text><path d="M39 7l2 6 6 2-6 2-2 6-2-6-6-2 6-2z" fill="#18C9A7"/>',
        "plus": '<path d="M24 10v28M10 24h28" stroke="#999999" stroke-width="4" stroke-linecap="round"/>',
        "save": '<path d="M9 7h24l6 6v28H9z" fill="#10B99A" opacity=".16"/><path d="M9 7h24l6 6v28H9zM16 7v11h14V7M17 31h14" fill="none" stroke="#10B99A" stroke-width="2.5"/>',
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


class Preview(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#FFFFFF; color:#222; border-radius:4px;")
        self.setText("Import Images")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def show_image(self, image):
        if image is None:
            self.setText("Import Images")
            self.setPixmap(QPixmap())
            return
        self.setText("")
        self.setPixmap(cv_to_pixmap(image, max(300, self.width() - 30), max(300, self.height() - 30)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "image") and self.image is not None:
            self.show_image(self.image)

    def set_image(self, image):
        self.image = image
        self.show_image(image)


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
        QPushButton#rotate { background:#F5F5F5; border-radius:8px; }
        QPushButton#tool { background:#F0F0F0; border-radius:9px; min-width:92px; min-height:80px; }
        QPushButton#tool:hover { background:#E8E8E8; }
        QPushButton#tool[selected='true'] { background:#D9F7F1; border:1px solid #10B99A; }
        QLabel#toolText { font-size:14px; }
        QPushButton#save { background:#10B99A; color:white; border-radius:8px; font-size:22px; min-width:180px; min-height:50px; }
        QPushButton#save:hover { background:#0DAE91; }
        QPushButton#add { background:#F0F0F0; border-radius:8px; }
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

        # Left: pages only, no filters or extra controls.
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

        # Center: image or the large + button.
        center = QFrame()
        center.setObjectName("center")
        cl = QVBoxLayout(center)
        cl.setContentsMargins(24, 24, 24, 24)

        self.preview = Preview()
        cl.addWidget(self.preview, 1)

        self.add_button = QPushButton()
        self.add_button.setObjectName("add")
        self.add_button.setIcon(make_icon("plus", size=64))
        self.add_button.setIconSize(QSize(64, 64))
        self.add_button.setFixedSize(125, 125)
        self.add_button.clicked.connect(self.open_image)
        # Put + exactly in the center over the empty preview.
        self.add_overlay = QFrame(self.preview)
        self.add_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        ol = QVBoxLayout(self.add_overlay)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(QLabel("Import Images"), 0, Qt.AlignCenter)
        ol.addWidget(self.add_button, 0, Qt.AlignCenter)
        self.add_overlay.setStyleSheet("background:transparent;")
        self.add_overlay.mousePressEvent = lambda e: self.open_image()
        self.add_overlay.raise_()
        self.add_overlay.setGeometry(0, 0, 1, 1)
        body.addWidget(center, 1)

        # Right: only the requested five controls.
        right = QFrame()
        right.setObjectName("right")
        right.setFixedWidth(270)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(28, 55, 28, 20)
        rv.setSpacing(20)

        rotations = QHBoxLayout()
        rotations.setSpacing(18)
        for kind, slot in (("left", lambda: self.rotate(-1)), ("right", lambda: self.rotate(1))):
            b = QPushButton()
            b.setObjectName("rotate")
            b.setFixedSize(70, 65)
            b.setIcon(make_icon(kind, size=40))
            b.setIconSize(QSize(40, 40))
            b.clicked.connect(slot)
            rotations.addWidget(b)
        rv.addLayout(rotations)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#E5E5E5;")
        rv.addWidget(line)

        self.original_btn = self.tool_button("original", "Original", "#1677FF", self.restore_original)
        self.magic_btn = self.tool_button("ai", "Magic Pro AI", "#00B89C", self.run_magic)
        self.original_btn.setProperty("selected", False)
        self.magic_btn.setProperty("selected", True)
        rv.addWidget(self.original_btn, 0, Qt.AlignHCenter)
        rv.addWidget(self.magic_btn, 0, Qt.AlignHCenter)
        rv.addStretch(1)

        body.addWidget(right)
        main.addLayout(body, 1)

        bottom = QFrame()
        bottom.setFixedHeight(70)
        bottom.setStyleSheet("border-top:1px solid #E5E6E8; background:#FFFFFF;")
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(25, 10, 18, 10)
        bl.addStretch()
        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("save")
        self.save_btn.clicked.connect(self.save_image)
        bl.addWidget(self.save_btn)
        main.addWidget(bottom)

        self.update_overlay()

    def tool_button(self, kind, text, color, slot):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        b = QPushButton()
        b.setObjectName("tool")
        b.setIcon(make_icon(kind, color, 52))
        b.setIconSize(QSize(52, 52))
        b.clicked.connect(slot)
        lay.addWidget(b)
        label = QLabel(text)
        label.setObjectName("toolText")
        label.setAlignment(Qt.AlignCenter)
        lay.addWidget(label)
        box.button = b
        return box

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_overlay()
        if self.original is not None:
            self.preview.show_image(self.magic if self.magic is not None else self.original)

    def update_overlay(self):
        if not hasattr(self, "add_overlay"):
            return
        if self.original is None:
            w = min(260, max(200, self.preview.width() // 2))
            h = 230
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

        # Show page thumbnail.
        pix = cv_to_pixmap(image, 105, 135)
        item = QListWidgetItem()
        item.setIcon(QIcon(pix))
        item.setSizeHint(QSize(115, 150))
        self.pages.clear()
        self.pages.addItem(item)
        self.pages.setCurrentRow(0)

        # Immediately apply Magic Pro AI as requested.
        self.run_magic()
        self.update_overlay()

    def select_page(self, row):
        if row < 0 or self.original is None:
            return
        self.restore_original()

    def restore_original(self):
        if self.original is None:
            return
        self.magic = None
        self.preview.set_image(self.original)
        self.original_btn.button.setProperty("selected", True)
        self.magic_btn.button.setProperty("selected", False)
        self.original_btn.button.style().unpolish(self.original_btn.button)
        self.original_btn.button.style().polish(self.original_btn.button)
        self.magic_btn.button.style().unpolish(self.magic_btn.button)
        self.magic_btn.button.style().polish(self.magic_btn.button)

    def run_magic(self):
        if self.original is None:
            self.open_image()
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            self.magic = magic_pro_ai(self.original, self.corners)
            self.preview.set_image(self.magic)
            self.magic_btn.button.setProperty("selected", True)
            self.original_btn.button.setProperty("selected", False)
            self.magic_btn.button.style().unpolish(self.magic_btn.button)
            self.magic_btn.button.style().polish(self.magic_btn.button)
            self.original_btn.button.style().unpolish(self.original_btn.button)
            self.original_btn.button.style().polish(self.original_btn.button)
        except Exception as exc:
            QMessageBox.warning(self, "Magic Pro AI", f"AI processing failed:\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def rotate(self, direction):
        if self.original is None:
            return
        if direction > 0:
            self.original = cv2.rotate(self.original, cv2.ROTATE_90_CLOCKWISE)
        else:
            self.original = cv2.rotate(self.original, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.corners = detect_document_corners(self.original)
        self.magic = None
        self.preview.set_image(self.original)
        self.update_overlay()
        self.run_magic()

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
            return
        QMessageBox.information(self, "Save", "Image saved successfully.")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = ScanPro()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
