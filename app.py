import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QProgressBar, QGraphicsDropShadowEffect, QLineEdit, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QColor, QFont, QIcon


# --- 1. Text Enhancement & Edge Processing ---

def enhance_text_clarity(image):
    """
    تحسين وضوح النصوص: إزالة الظلال وتنقية خلفية المستند وزيادة حدة الخطوط
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # تصحيح الإضاءة وإزالة الظلال خلف النصوص
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    # زيادة حدة ووضوح الكلمات والخطوط
    gaussian = cv2.GaussianBlur(norm_img, (0, 0), 3.0)
    sharpened = cv2.addWeighted(norm_img, 1.8, gaussian, -0.8, 0)
    
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


# --- 2. Custom Animated Plus Button ---

class AnimatedPlusButton(QPushButton):
    """
    زر إضافة تفاعلي يتغير حجمه ولونه عند مرور مؤشر الفأرة
    """
    def __init__(self, parent=None):
        super().__init__("+", parent)
        self.setFont(QFont("Segoe UI", 36, QFont.Bold))
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(90, 90)
        self.setStyleSheet("""
            QPushButton {
                background-color: #00B89C;
                color: #FFFFFF;
                border-radius: 45px;
                border: none;
            }
        """)
        
        # إضافة تأثير الظل بطريقة صحيحة متوافقة مع PySide6
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 184, 156, 100))
        shadow.setOffset(0, 4)  # تصحيح الخطأ هنا
        self.setGraphicsEffect(shadow)

    def enterEvent(self, event):
        self.setStyleSheet("""
            QPushButton {
                background-color: #008f79;
                color: #FFFFFF;
                border-radius: 45px;
                border: 2px solid #00F5D4;
            }
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet("""
            QPushButton {
                background-color: #00B89C;
                color: #FFFFFF;
                border-radius: 45px;
                border: none;
            }
        """)
        super().leaveEvent(event)


# --- 3. Processing Thread ---

class BatchProcessorThread(QThread):
    progress = Signal(int, int, str)
    finished = Signal(list)

    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths

    def run(self):
        processed_results = []
        total = len(self.file_paths)

        for idx, path in enumerate(self.file_paths):
            img = cv2.imread(path)
            if img is not None:
                enhanced = enhance_text_clarity(img)
                processed_results.append((path, enhanced))
            self.progress.emit(idx + 1, total, os.path.basename(path))

        self.finished.emit(processed_results)


# --- 4. Main Application Window ---

class ScanProAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scan Pro AI")
        self.resize(1100, 750)
        self.loaded_files = []
        self.processed_data = []

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #F4F6F9;")
        self.setCentralWidget(main_widget)

        layout = QHBoxLayout(main_widget)

        # الجانب الأيسر - منطقة العرض والتحكم
        left_panel = QVBoxLayout()
        
        self.center_area = QVBoxLayout()
        self.center_area.setAlignment(Qt.AlignCenter)

        self.btn_add = AnimatedPlusButton()
        self.btn_add.clicked.connect(self.open_files)
        self.center_area.addWidget(self.btn_add, alignment=Qt.AlignCenter)

        self.lbl_status = QLabel("إضغط على الزر أعلاه لاختيار الملفات أو المستندات")
        self.lbl_status.setFont(QFont("Segoe UI", 11))
        self.lbl_status.setStyleSheet("color: #7F8C8D; margin-top: 10px;")
        self.center_area.addWidget(self.lbl_status, alignment=Qt.AlignCenter)

        left_panel.addLayout(self.center_area)

        # قائمة الملفات مع خاصية التحديد التلقائي عند مركب الفأرة (Hover Selection)
        self.file_list = QListWidget()
        self.file_list.setMouseTracking(True)
        self.file_list.itemEntered.connect(self.on_item_hovered)
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #F0F0F0;
            }
            QListWidget::item:hover {
                background-color: #E6F7F5;
                color: #00B89C;
            }
            QListWidget::item:selected {
                background-color: #00B89C;
                color: #FFFFFF;
                font-weight: bold;
            }
        """)
        left_panel.addWidget(self.file_list)

        # شريط التقدم
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #E0E0E0;
                height: 8px;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #00B89C;
                border-radius: 4px;
            }
        """)
        left_panel.addWidget(self.progress_bar)

        layout.addLayout(left_panel, stretch=3)

        # الجانب الأيمن - الخيارات والتنفيذ
        right_panel = QVBoxLayout()
        right_panel.setAlignment(Qt.AlignTop)

        lbl_title = QLabel("Scan Pro AI")
        lbl_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        lbl_title.setStyleSheet("color: #2C3E50; margin-bottom: 20px;")
        right_panel.addWidget(lbl_title)

        self.btn_process = QPushButton("Magic Pro AI (تحسين النص)")
        self.btn_process.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_process.setCursor(Qt.PointingHandCursor)
        self.btn_process.setStyleSheet("""
            QPushButton {
                background-color: #00B89C;
                color: white;
                padding: 12px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #008f79;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.process_batch)
        right_panel.addWidget(self.btn_process)

        # خيارات التصدير وتسمية الصور
        export_box = QVBoxLayout()
        export_box.setContentsMargins(0, 20, 0, 0)

        lbl_prefix = QLabel("بادئة اسم الصور (Prefix):")
        lbl_prefix.setFont(QFont("Segoe UI", 10))
        export_box.addWidget(lbl_prefix)

        self.txt_prefix = QLineEdit("Doc_Page")
        self.txt_prefix.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #BDC3C7;
                border-radius: 4px;
                background-color: white;
            }
        """)
        export_box.addWidget(self.txt_prefix)

        self.btn_save = QPushButton("حفظ الصور المتسلسلة")
        self.btn_save.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: white;
                padding: 12px;
                border-radius: 6px;
                border: none;
                margin-top: 15px;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_files)
        export_box.addWidget(self.btn_save)

        right_panel.addLayout(export_box)
        layout.addLayout(right_panel, stretch=1)

    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "اختر الصور/المستندات", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if files:
            self.loaded_files = files
            self.file_list.clear()
            for f in files:
                self.file_list.addItem(os.path.basename(f))
            self.lbl_status.setText(f"تم اختيار {len(files)} ملفات. قم بالوقوف على أي صورة لتحديدها.")
            self.btn_process.setEnabled(True)

    def on_item_hovered(self, item):
        """
        تحديد الصورة في القائمة تلقائياً بمجرد الوقوف عليها بمؤشر الفأرة
        """
        if item:
            self.file_list.setCurrentItem(item)
            self.lbl_status.setText(f"الصورة المحددة حالياً: {item.text()}")

    def process_batch(self):
        if not self.loaded_files:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_process.setEnabled(False)

        self.thread = BatchProcessorThread(self.loaded_files)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.on_processing_finished)
        self.thread.start()

    def update_progress(self, current, total, filename):
        val = int((current / total) * 100)
        self.progress_bar.setValue(val)
        self.lbl_status.setText(f"جاري معالجة: {filename} ({current}/{total})")

    def on_processing_finished(self, results):
        self.processed_data = results
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("اكتملت المعالجة بنجاح! يمكنك الآن حفظ الصور المتسلسلة.")
        self.btn_save.setEnabled(True)
        self.btn_process.setEnabled(True)

    def save_files(self):
        if not self.processed_data:
            return

        output_dir = QFileDialog.getExistingDirectory(self, "اختر مجلد الحفظ")
        if not output_dir:
            return

        prefix = self.txt_prefix.text().strip() or "Document"

        # حفظ كصور بأسماء متسلسلة (مثال: Doc_Page_001.png, Doc_Page_002.png)
        for idx, (_, img_bgr) in enumerate(self.processed_data, start=1):
            file_name = f"{prefix}_{idx:03d}.png"
            full_path = os.path.join(output_dir, file_name)
            cv2.imwrite(full_path, img_bgr)

        QMessageBox.information(self, "تم الحفظ", f"تم حفظ {len(self.processed_data)} صورة بأسماء وارقام متسلسلة بنجاح!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScanProAIApp()
    window.show()
    sys.exit(app.exec())
