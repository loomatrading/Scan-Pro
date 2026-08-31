import sys
import os
import ctypes
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QSize, QPointF, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QImage, QPixmap, QPainter, QIcon, QPen, QBrush, QColor, QKeySequence, QShortcut, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QFrame,
    QListWidget, QListWidgetItem, QSizePolicy, QGraphicsDropShadowEffect
)


APP_NAME = "Scan Pro AI"
MODELS_DIR = Path(__file__).resolve().parent / "models"
EDSR_URL = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb"
EDSR_FILE = MODELS_DIR / "EDSR_x2.pb"


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # 1. استخدام التوزيع الظلي لعزل الخلفية مع المحافظة على دقة الحروف الدقيقة
    bg = cv2.GaussianBlur(gray, (51, 51), 0)
    
    # 2. قسمة الصورة على الخلفية لتبييض المساحات الفارغة دون تضخيم الخط
    norm = cv2.divide(gray, bg, scale=255.0)

    # 3. توضيح معالم النص الرفيع عبر تمديد النطاق الديناميكي (Contrast Stretch)
    norm = cv2.normalize(norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 4. تفتيح إضافي للرماديات الضعيفة في الخلفية لضمان البياض التام
    _, thresh = cv2.threshold(norm, 230, 255, cv2.THRESH_TOZERO)

    # 5. تنظيف الحواف الخارجية
    h, w = thresh.shape
    margin_x = max(1, int(w * 0.012))
    margin_y = max(1, int(h * 0.012))
    
    thresh[:margin_y, :] = 255
    thresh[-margin_y:, :] = 255
    thresh[:, :margin_x] = 255
    thresh[:, -margin_x:] = 255

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


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
        "original": f'<rect x="10" y="6" width="28" height="36" rx="3" fill="{color}" opacity=".12"/><rect x="10" y="6" width="28" height="36" rx="3" fill="none" stroke="{color}" stroke-width="2.5"/><path d="M16 16h16M16 23h16M16 30h12M16 37h8" stroke="{color}" stroke-width="2.5"/>',
        "ai": '<text x="4" y="36" font-family="Arial" font-size="30" font-weight="700" fill="#00B89C">AI</text><path d="M39 7l2 6 6 2-6 2-2 6-2-6-6-2 6-2z" fill="#18C9A7"/>',
        "trash": f'<path d="M12 14h24M18 14V10h12v4M15 14v22a2 2 0 002 2h14a2 2 0 002-2V14" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><path d="M20 20v10M28 20v10" stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
    }
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48">{icons.get(kind, "")}</svg>'


def make_icon(kind, color="#111111", size=46):
    from PySide6.QtSvg import QSvgRenderer
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    QSvgRenderer(svg_icon(kind, color).encode("utf-8")).render(painter)
    painter.end()
    return QIcon(pix)


def get_image_icon(filename, fallback_kind, color="#111111", size=46):
    full_path = resource_path(filename)
    if os.path.exists(full_path):
        return QIcon(full_path)
    return make_icon(fallback_kind, color, size)


class AnimatedPlusButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__("+", parent)
        self.setFont(QFont("Segoe UI", 42, QFont.Bold))
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(130, 130)
        
        self.default_style = """
            QPushButton {
                background-color: #EFEFEF;
                color: #777777;
                border-radius: 22px;
                border: 2px dashed #CCCCCC;
                padding-bottom: 8px;
                text-align: center;
            }
        """
        self.hover_style = """
            QPushButton {
                background-color: #E6F7F5;
                color: #00B89C;
                border-radius: 22px;
                border: 2px solid #00B89C;
                padding-bottom: 8px;
                text-align: center;
            }
        """
        self.setStyleSheet(self.default_style)
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(16)
        shadow.setColor(QColor(0, 0, 0, 25))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

    def enterEvent(self, event):
        self.setStyleSheet(self.hover_style)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self.default_style)
        super().leaveEvent(event)


class ClickableOverlayFrame(QFrame):
    def __init__(self, click_callback, parent=None):
        super().__init__(parent)
        self.click_callback = click_callback

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.click_callback()
            event.accept()
        else:
            super().mousePressEvent(event)


class InteractivePreview(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#FFFFFF; color:#222; border-radius:4px;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image = None
        self.base_image = None
        self.corners = None
        self.active_handle = -1
        self.eraser_active = False
        self.brush_size = 24
        self.zoom_factor = 1.0
        self.history = []
        self.mouse_pos = QPointF(-100, -100)
        self.corner_changed_callback = None
        self.image_edited_callback = None
        self.click_to_open_callback = None
        self.setMouseTracking(True)

    def save_undo_state(self):
        if self.image is not None:
            self.history.append(self.image.copy())
            if len(self.history) > 15:
                self.history.pop(0)

    def undo(self):
        if self.history:
            self.image = self.history.pop()
            self.update()
            if self.image_edited_callback:
                self.image_edited_callback(self.image)

    def set_data(self, image, corners=None, base_image=None):
        if image is not None and self.image is not None:
            self.save_undo_state()
        
        self.image = image.copy() if image is not None else None
        
        if base_image is not None:
            self.base_image = base_image
        elif self.base_image is None:
            self.base_image = image

        if corners is not None:
            self.corners = corners.copy()
            
        self.update()
        QApplication.processEvents()

    def set_eraser_mode(self, active):
        self.eraser_active = active
        if active:
            self.setCursor(Qt.BlankCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def get_disp_rect(self):
        if self.image is None:
            return None
        h, w = self.image.shape[:2]
        pw, ph = self.width(), self.height()
        scale = min(pw / w, ph / h) * self.zoom_factor
        nw, nh = int(w * scale), int(h * scale)
        ox = (pw - nw) // 2
        oy = (ph - nh) // 2
        return ox, oy, nw, nh, scale

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_factor = min(self.zoom_factor * 1.15, 5.0)
            else:
                self.zoom_factor = max(self.zoom_factor / 1.15, 0.5)
            self.update()
            event.accept()
        else:
            super().wheelEvent(event)

    def get_all_handles(self):
        if self.corners is None:
            return []
        pts = list(self.corners)
        edges = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            edges.append([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0])
        return pts + edges

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image is None:
            return

        rect_info = self.get_disp_rect()
        if not rect_info:
            return
        ox, oy, nw, nh, scale = rect_info

        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(nw, nh, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        painter = QPainter(self)
        painter.drawPixmap(ox, oy, pix)

        if self.base_image is not None and self.corners is not None and not self.eraser_active:
            bh, bw = self.base_image.shape[:2]
            b_scale = min(self.width() / bw, self.height() / bh) * self.zoom_factor
            box_x = (self.width() - int(bw * b_scale)) // 2
            box_y = (self.height() - int(bh * b_scale)) // 2

            handles = self.get_all_handles()
            disp_pts = []
            for h_pt in handles:
                x = box_x + h_pt[0] * b_scale
                y = box_y + h_pt[1] * b_scale
                disp_pts.append(QPointF(x, y))

            pen = QPen(QColor("#10B99A"), 2, Qt.DashLine)
            painter.setPen(pen)
            for i in range(4):
                painter.drawLine(disp_pts[i], disp_pts[(i + 1) % 4])

            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.setBrush(QBrush(QColor("#10B99A")))
            for p in disp_pts[:4]:
                painter.drawEllipse(p, 7, 7)

            painter.setBrush(QBrush(QColor("#1677FF")))
            for p in disp_pts[4:]:
                painter.drawRect(p.x() - 5, p.y() - 5, 10, 10)

        if self.eraser_active:
            box_sz = int(self.brush_size * scale * 2)
            half = box_sz / 2.0
            x = self.mouse_pos.x() - half
            y = self.mouse_pos.y() - half

            painter.setPen(QPen(QColor(0, 0, 0, 180), 2, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x - 1, y - 1, box_sz + 2, box_sz + 2)

            painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.DashLine))
            painter.drawRect(x, y, box_sz, box_sz)

        painter.end()

    def erase_at(self, pos):
        if self.image is None or not self.eraser_active:
            return
        rect_info = self.get_disp_rect()
        if not rect_info:
            return
        ox, oy, _, _, scale = rect_info

        img_x = int((pos.x() - ox) / scale)
        img_y = int((pos.y() - oy) / scale)

        h, w = self.image.shape[:2]
        if 0 <= img_x < w and 0 <= img_y < h:
            self.save_undo_state()
            cv2.rectangle(
                self.image,
                (img_x - self.brush_size, img_y - self.brush_size),
                (img_x + self.brush_size, img_y + self.brush_size),
                (255, 255, 255), -1
            )
            self.update()
            if self.image_edited_callback:
                self.image_edited_callback(self.image)

    def mousePressEvent(self, event):
        if self.image is None and self.click_to_open_callback:
            self.click_to_open_callback()
            return

        pos = event.position()
        if self.eraser_active:
            self.erase_at(pos)
            return

        if self.base_image is None or self.corners is None:
            return

        bh, bw = self.base_image.shape[:2]
        b_scale = min(self.width() / bw, self.height() / bh) * self.zoom_factor
        box_x = (self.width() - int(bw * b_scale)) // 2
        box_y = (self.height() - int(bh * b_scale)) // 2

        handles = self.get_all_handles()
        for i, h_pt in enumerate(handles):
            cx = box_x + h_pt[0] * b_scale
            cy = box_y + h_pt[1] * b_scale
            if (pos.x() - cx) ** 2 + (pos.y() - cy) ** 2 <= 400:
                self.active_handle = i
                break

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.position()
        if self.eraser_active:
            if event.buttons() & Qt.LeftButton:
                self.erase_at(self.mouse_pos)
            self.update()
            return

        if self.active_handle != -1 and self.corners is not None and self.base_image is not None:
            bh, bw = self.base_image.shape[:2]
            b_scale = min(self.width() / bw, self.height() / bh) * self.zoom_factor
            box_x = (self.width() - int(bw * b_scale)) // 2
            box_y = (self.height() - int(bh * b_scale)) // 2

            x = max(0, min((self.mouse_pos.x() - box_x) / b_scale, bw))
            y = max(0, min((self.mouse_pos.y() - box_y) / b_scale, bh))

            if self.active_handle < 4:
                self.corners[self.active_handle] = [x, y]
            else:
                edge_idx = self.active_handle - 4
                c1 = edge_idx
                c2 = (edge_idx + 1) % 4
                old_mid_x = (self.corners[c1][0] + self.corners[c2][0]) / 2.0
                old_mid_y = (self.corners[c1][1] + self.corners[c2][1]) / 2.0
                dx = x - old_mid_x
                dy = y - old_mid_y

                self.corners[c1] = [max(0, min(self.corners[c1][0] + dx, bw)), max(0, min(self.corners[c1][1] + dy, bh))]
                self.corners[c2] = [max(0, min(self.corners[c2][0] + dx, bw)), max(0, min(self.corners[c2][1] + dy, bh))]

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
        self.setMinimumSize(1000, 650)

        self.original = None
        self.magic = None
        self.corners = None
        self.save_counter = 1

        self.build_ui()
        self.setup_shortcuts()

    def setup_shortcuts(self):
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_action)

    def undo_action(self):
        self.preview.undo()

    def build_ui(self):
        self.setStyleSheet("""
        QMainWindow, QWidget { background:#FFFFFF; color:#202020; font-family:'Segoe UI'; }
        #left { border-right:1px solid #E4E5E7; background:#FFFFFF; }
        #center { background:#F1F2F4; }
        #right { border-left:1px solid #E4E5E7; background:#FFFFFF; }
        #topline { border-bottom:1px solid #E5E6E8; }
        QPushButton { border:none; }
        QPushButton#rotate { background:transparent; border-radius:10px; min-width:76px; min-height:76px; }
        QPushButton#rotate:hover { background:#F0E6FF; }
        QPushButton#tool { background:#F0F0F0; border-radius:9px; min-width:85px; min-height:72px; }
        QPushButton#tool:hover { background:#E8E8E8; }
        QPushButton#tool[selected='true'] { background:#D9F7F1; border:1px solid #10B99A; }
        QPushButton#delete { background:#F0F0F0; border-radius:9px; min-width:85px; min-height:72px; }
        QPushButton#delete:hover { background:#FFE6E6; }
        QLabel#toolText { font-size:13px; }
        QPushButton#save { background:#00B89C; color:#FFFFFF; border-radius:8px; font-size:18px; font-weight:bold; min-width:140px; min-height:42px; padding:4px 16px; }
        QPushButton#save:hover { background:#019D85; }
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

        center = QFrame()
        center.setObjectName("center")
        cl = QVBoxLayout(center)
        cl.setContentsMargins(20, 20, 20, 20)

        self.preview = InteractivePreview()
        self.preview.corner_changed_callback = self.on_corners_updated
        self.preview.image_edited_callback = self.on_image_edited
        self.preview.click_to_open_callback = self.open_image
        cl.addWidget(self.preview, 1)

        self.add_button = AnimatedPlusButton()
        self.add_button.clicked.connect(self.open_image)

        self.add_overlay = ClickableOverlayFrame(self.open_image, self.preview)
        self.add_overlay.setCursor(Qt.PointingHandCursor)
        ol = QVBoxLayout(self.add_overlay)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(self.add_button, 0, Qt.AlignCenter)
        self.add_overlay.setStyleSheet("background:transparent;")
        self.add_overlay.raise_()
        body.addWidget(center, 1)

        right = QFrame()
        right.setObjectName("right")
        right.setFixedWidth(240)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(20, 20, 20, 20)
        rv.setSpacing(12)

        self.rotate_btn = QPushButton()
        self.rotate_btn.setObjectName("rotate")
        self.rotate_btn.setFixedSize(76, 76)
        self.rotate_btn.setIcon(get_image_icon("rotate.png", "rotate_single", size=64))
        self.rotate_btn.setIconSize(QSize(64, 64))
        self.rotate_btn.clicked.connect(lambda: self.rotate(1))
        rv.addWidget(self.rotate_btn, 0, Qt.AlignHCenter)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#E5E5E5;")
        rv.addWidget(line)

        self.original_btn = self.tool_button("original", "Original", "#1677FF", self.restore_original)
        self.magic_btn = self.tool_button("ai", "Magic Pro AI", "#00B89C", self.run_magic)
        self.cleaner_btn = self.tool_button_custom("cleaner.png", "Cleaner", "#0088FF", self.toggle_cleaner)
        self.delete_btn = self.tool_button("trash", "", "#D32F2F", self.delete_image, is_delete=True)

        self.original_btn.button.setProperty("selected", False)
        self.magic_btn.button.setProperty("selected", True)

        rv.addWidget(self.original_btn, 0, Qt.AlignHCenter)
        rv.addWidget(self.magic_btn, 0, Qt.AlignHCenter)
        rv.addWidget(self.cleaner_btn, 0, Qt.AlignHCenter)
        rv.addWidget(self.delete_btn, 0, Qt.AlignHCenter)

        rv.addStretch(1)

        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("save")
        self.save_btn.clicked.connect(self.save_image)
        rv.addWidget(self.save_btn, 0, Qt.AlignHCenter)

        body.addWidget(right)
        main.addLayout(body, 1)

        self.update_overlay()

    def tool_button(self, kind, text, color, slot, is_delete=False):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        b = QPushButton()
        b.setObjectName("delete" if is_delete else "tool")
        b.setIcon(make_icon(kind, color, 46))
        b.setIconSize(QSize(46, 46))
        b.clicked.connect(slot)
        lay.addWidget(b)
        if text:
            label = QLabel(text)
            label.setObjectName("toolText")
            label.setAlignment(Qt.AlignCenter)
            lay.addWidget(label)
        box.button = b
        return box

    def tool_button_custom(self, img_file, text, color, slot):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        b = QPushButton()
        b.setObjectName("tool")
        b.setIcon(get_image_icon(img_file, "cleaner", color, 46))
        b.setIconSize(QSize(46, 46))
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

    def toggle_cleaner(self):
        if self.original is None:
            return
        is_active = not self.preview.eraser_active
        self.preview.set_eraser_mode(is_active)
        self.cleaner_btn.button.setProperty("selected", is_active)
        self.update_button_styles()

    def on_corners_updated(self, updated_corners):
        self.corners = updated_corners
        if self.magic_btn.button.property("selected"):
            self.run_magic()

    def on_image_edited(self, edited_img):
        if self.magic_btn.button.property("selected"):
            self.magic = edited_img.copy()

    def select_page(self, row):
        if row < 0 or self.original is None:
            return
        self.restore_original()

    def restore_original(self):
        if self.original is None:
            return
        self.preview.set_eraser_mode(False)
        self.preview.set_data(self.original, self.corners, base_image=self.original)
        self.original_btn.button.setProperty("selected", True)
        self.magic_btn.button.setProperty("selected", False)
        self.cleaner_btn.button.setProperty("selected", False)
        self.update_button_styles()

    def run_magic(self):
        if self.original is None:
            self.open_image()
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            self.magic = magic_pro_ai(self.original, self.corners)
            self.preview.set_eraser_mode(False)
            self.preview.set_data(self.magic, self.corners, base_image=self.original)
            self.magic_btn.button.setProperty("selected", True)
            self.original_btn.button.setProperty("selected", False)
            self.cleaner_btn.button.setProperty("selected", False)
            self.update_button_styles()
            self.preview.repaint()
        except Exception as exc:
            QMessageBox.warning(self, "Magic Pro AI", f"AI processing failed:\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def update_button_styles(self):
        for btn in (self.original_btn.button, self.magic_btn.button, self.cleaner_btn.button):
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
        self.preview.set_eraser_mode(False)
        self.preview.set_data(None, None, None)
        self.update_overlay()

    def save_image(self):
        if self.original is None:
            QMessageBox.information(self, "Save", "Import an image first.")
            return

        image = self.preview.image if self.preview.image is not None else self.original

        while True:
            default_name = f"ScanPro_{self.save_counter}.jpg"
            default_path = os.path.join(desktop_path(), default_name)
            if not os.path.exists(default_path):
                break
            self.save_counter += 1

        fn, _ = QFileDialog.getSaveFileName(
            self, "Save Image", default_path,
            "JPEG Image (*.jpg *.jpeg);;PNG Image (*.png)"
        )
        if not fn:
            return

        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            fn += ".jpg"

        ok = cv2.imwrite(fn, image, [cv2.IMWRITE_JPEG_QUALITY, 98] if fn.lower().endswith((".jpg", ".jpeg")) else [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if ok:
            self.save_counter += 1
        else:
            QMessageBox.critical(self, "Save", "Could not save the image.")


def main():
    try:
        myappid = 'scanpro.ai.scanner.app.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    
    ico_path = resource_path("app_icon.ico")
    if os.path.exists(ico_path):
        app_icon = QIcon(ico_path)
        app.setWindowIcon(app_icon)

    window = ScanPro()
    if os.path.exists(ico_path):
        window.setWindowIcon(QIcon(ico_path))
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
