import os
import sys
import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout, QFrame

APP_NAME = "ScanPro"
MODEL_NAME = "EDSR_x2.pb"


def resource_path(*parts):
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)


def desktop_path():
    path = os.path.join(os.path.expanduser("~"), "Desktop")
    os.makedirs(path, exist_ok=True)
    return path


def order_points(points):
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def detect_document_corners(image):
    h, w = image.shape[:2]
    scale = min(1.0, 1600.0 / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    candidates = []

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    c, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(c)

    edges = cv2.Canny(gray, 35, 130)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    c, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(c)

    area_img = small.shape[0] * small.shape[1]
    best = None
    best_score = -1
    for contour in candidates:
        area = cv2.contourArea(contour)
        if area < area_img * 0.18:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        rect_area = cv2.contourArea(pts)
        if rect_area <= 0:
            continue
        rectangularity = area / rect_area
        if rectangularity < 0.72:
            continue
        score = (area / area_img) * 3.0 + rectangularity
        if score > best_score:
            best_score = score
            best = pts

    if best is not None:
        if scale < 1:
            best /= scale
        return order_points(best)

    mx, my = w * .02, h * .02
    return np.array([[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]], dtype=np.float32)


def perspective_transform(image, corners):
    tl, tr, br, bl = order_points(corners)
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width = max(width, 300)
    height = max(height, 300)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def document_enhance(img):
    """Scanner-like automatic enhancement: illumination, contrast, denoise and text detail."""
    result = img.copy()
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    bg = np.maximum(bg, 1)
    corrected = cv2.divide(gray, bg, scale=255)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = corrected
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21)
    blur = cv2.GaussianBlur(result, (0, 0), 1.0)
    return cv2.addWeighted(result, 1.12, blur, -0.12, 0)


def load_ai_model():
    """Load the real EDSR x2 AI model bundled with the application."""
    model_path = resource_path("models", MODEL_NAME)
    if not os.path.exists(model_path):
        return None
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("edsr", 2)
        return sr
    except Exception:
        return None


_AI_MODEL = None
_AI_MODEL_TRIED = False


def ai_upscale(img):
    global _AI_MODEL, _AI_MODEL_TRIED
    if not _AI_MODEL_TRIED:
        _AI_MODEL_TRIED = True
        _AI_MODEL = load_ai_model()
    if _AI_MODEL is None:
        return img
    try:
        # Keep very large scans manageable. EDSR is applied to the document crop.
        h, w = img.shape[:2]
        if max(h, w) > 2600:
            scale = 2600.0 / max(h, w)
            work = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            up = _AI_MODEL.upsample(work)
            target_w, target_h = int(w * 2), int(h * 2)
            return cv2.resize(up, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        return _AI_MODEL.upsample(img)
    except Exception:
        return img


def magic_pro_ai(image, corners):
    cropped = perspective_transform(image, corners)
    enhanced = document_enhance(cropped)
    # AI super-resolution is applied after scanner correction.
    return ai_upscale(enhanced)


def cv_to_pixmap(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def svg_icon(kind, color="#111111"):
    paths = {
        "plus": f'<path d="M24 8v32M8 24h32" stroke="{color}" stroke-width="4" stroke-linecap="round"/>',
        "left": f'<path d="M31 12a14 14 0 1 0 3 22" fill="none" stroke="{color}" stroke-width="3"/><path d="M12 9v10h10" fill="none" stroke="{color}" stroke-width="3"/>',
        "right": f'<path d="M17 12a14 14 0 1 1-3 22" fill="none" stroke="{color}" stroke-width="3"/><path d="M36 9v10H26" fill="none" stroke="{color}" stroke-width="3"/>',
        "original": f'<rect x="10" y="7" width="28" height="34" rx="3" fill="none" stroke="{color}" stroke-width="2.5"/><path d="M16 17h16M16 24h16M16 31h11" stroke="{color}" stroke-width="3"/>',
        "ai": '<text x="4" y="36" font-family="Arial" font-size="30" font-weight="700" fill="#00B89C">AI</text><path d="M38 6l2 6 6 2-6 2-2 6-2-6-6-2 6-2z" fill="#18C9A7"/>',
        "save": f'<path d="M9 7h24l6 6v28H9z" fill="{color}" opacity=".12"/><path d="M9 7h24l6 6v28H9zM16 7v12h17V7M16 41V28h17v13" fill="none" stroke="{color}" stroke-width="2.5"/>',
    }
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">{paths[kind]}</svg>'


def make_icon(kind, color="#111111", size=42):
    from PySide6.QtSvg import QSvgRenderer
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    QSvgRenderer(svg_icon(kind, color).encode("utf-8")).render(painter)
    painter.end()
    return QIcon(pix)


class ImageCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.image = None
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#F1F1F1;")

    def set_image(self, image):
        self.image = image
        self.refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()

    def refresh(self):
        if self.image is None:
            self.setPixmap(QPixmap())
            return
        pix = cv_to_pixmap(self.image)
        self.setPixmap(pix.scaled(self.size() - QSize(40, 40), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class ScanPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScanPro")
        self.resize(1500, 900)
        self.setMinimumSize(1000, 650)
        self.original = None
        self.processed = None
        self.corners = None
        self.build_ui()

    def build_ui(self):
        self.setStyleSheet("""
            QMainWindow { background:#FFFFFF; }
            #work { background:#F1F1F1; }
            #tools { background:#FFFFFF; border-left:1px solid #DDDDDD; }
            #iconButton { background:#F3F3F3; border-radius:9px; }
            #iconButton:hover { background:#E9E9E9; }
            #tool { background:#EEEEEE; border-radius:10px; }
            #tool[selected="true"] { background:#DFFAF4; border:1.5px solid #10BFA1; }
            #toolLabel { font-size:15px; color:#333333; }
            #magicLabel { font-size:15px; color:#00B89C; }
            #save { background:#10B99A; color:#FFFFFF; border-radius:8px; font-size:22px; min-height:50px; }
            #save:hover { background:#0EAA8B; }
            #empty { color:#222222; font-size:21px; }
        """)

        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        work = QFrame()
        work.setObjectName("work")
        wl = QVBoxLayout(work)
        wl.setContentsMargins(20, 20, 20, 20)

        self.canvas = ImageCanvas()
        wl.addWidget(self.canvas, 1)

        empty = QWidget()
        el = QVBoxLayout(empty)
        el.setAlignment(Qt.AlignCenter)
        self.empty_label = QLabel("Import Images")
        self.empty_label.setObjectName("empty")
        self.empty_label.setAlignment(Qt.AlignCenter)
        el.addWidget(self.empty_label)
        self.add_button = QPushButton()
        self.add_button.setIcon(make_icon("plus", "#999999", 68))
        self.add_button.setIconSize(QSize(68, 68))
        self.add_button.setFixedSize(110, 110)
        self.add_button.setStyleSheet("QPushButton{background:#EEEEEE;border-radius:8px;} QPushButton:hover{background:#E5E5E5;}")
        self.add_button.clicked.connect(self.import_file)
        el.addWidget(self.add_button, 0, Qt.AlignHCenter)
        wl.addWidget(empty, 0, Qt.AlignCenter)
        self.empty_widget = empty

        body.addWidget(work, 1)

        tools = QFrame()
        tools.setObjectName("tools")
        tools.setFixedWidth(260)
        tv = QVBoxLayout(tools)
        tv.setContentsMargins(20, 20, 20, 16)
        tv.setSpacing(14)

        rot = QHBoxLayout()
        left = self.make_icon_button("left", lambda: self.rotate(-1))
        right = self.make_icon_button("right", lambda: self.rotate(1))
        rot.addWidget(left)
        rot.addWidget(right)
        tv.addLayout(rot)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#E5E5E5;")
        tv.addWidget(line)

        self.original_button = self.make_tool("original", "Original", "#1677FF", self.show_original)
        self.magic_button = self.make_tool("ai", "Magic Pro AI", "#00B89C", self.run_magic)
        self.magic_button.setProperty("selected", True)
        self.magic_button.style().unpolish(self.magic_button)
        self.magic_button.style().polish(self.magic_button)
        tv.addWidget(self.original_button)
        tv.addWidget(self.magic_button)
        tv.addStretch(1)

        save = QPushButton("Save")
        save.setObjectName("save")
        save.setIcon(make_icon("save", "#FFFFFF", 32))
        save.setIconSize(QSize(32, 32))
        save.clicked.connect(self.save_image)
        tv.addWidget(save)
        body.addWidget(tools)

        main.addLayout(body, 1)

    def make_icon_button(self, kind, slot):
        b = QPushButton()
        b.setObjectName("iconButton")
        b.setIcon(make_icon(kind, "#111111", 36))
        b.setIconSize(QSize(36, 36))
        b.setFixedSize(70, 60)
        b.clicked.connect(slot)
        return b

    def make_tool(self, kind, text, color, slot):
        b = QPushButton()
        b.setObjectName("tool")
        b.setIcon(make_icon(kind, color, 50))
        b.setIconSize(QSize(50, 50))
        b.setText("\n" + text)
        b.setFixedHeight(115)
        b.setStyleSheet("QPushButton{text-align:center;font-size:15px;padding:8px;} QPushButton:hover{background:#E8E8E8;}")
        b.clicked.connect(slot)
        return b

    def import_file(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Import Images", desktop_path(), "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if not fn:
            return
        image = cv2.imread(fn, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.critical(self, "Error", "Could not open the image.")
            return
        self.original = image
        self.corners = detect_document_corners(image)
        self.empty_widget.hide()
        self.process_magic()

    def process_magic(self):
        if self.original is None or self.corners is None:
            return
        self.magic_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.processed = magic_pro_ai(self.original, self.corners)
            self.canvas.set_image(self.processed)
        finally:
            QApplication.restoreOverrideCursor()
            self.magic_button.setEnabled(True)

    def run_magic(self):
        self.process_magic()

    def show_original(self):
        if self.original is None:
            return
        self.canvas.set_image(self.original)

    def rotate(self, direction):
        if self.original is None:
            return
        self.original = cv2.rotate(self.original, cv2.ROTATE_90_CLOCKWISE if direction > 0 else cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.corners = detect_document_corners(self.original)
        self.process_magic()

    def save_image(self):
        if self.processed is None:
            QMessageBox.information(self, "Save", "Import an image first.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save Image", os.path.join(desktop_path(), "ScanPro.jpg"), "JPEG Image (*.jpg *.jpeg);;PNG Image (*.png)")
        if not fn:
            return
        if fn.lower().endswith(".png"):
            ok = cv2.imwrite(fn, self.processed, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            if not fn.lower().endswith((".jpg", ".jpeg")):
                fn += ".jpg"
            ok = cv2.imwrite(fn, self.processed, [cv2.IMWRITE_JPEG_QUALITY, 98])
        if ok:
            QMessageBox.information(self, "Saved", "Image saved successfully on Desktop.")
        else:
            QMessageBox.critical(self, "Error", "Could not save the image.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ScanPro()
    win.show()
    sys.exit(app.exec())
