from PyQt5.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QMessageBox
)
from labelme.logger import logger
class CustomDialog(QDialog):
    def __init__(self, title="Custom Dialog", fields=None, parent=None, show_help=False):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        self.input_fields = {}  
        self.layout = QVBoxLayout()
        self.help_content = "" 
        self.show_help_button = show_help  

        if fields is not None:
            self._create_fields(fields)
        
        button_layout = QHBoxLayout()
        submit_button = QPushButton("Submit")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(submit_button)
        button_layout.addWidget(cancel_button)

        if self.show_help_button:
            help_button = QPushButton("Help")
            help_button.clicked.connect(self.show_help_dialog)
            button_layout.addWidget(help_button)

        self.layout.addLayout(button_layout)
        self.setLayout(self.layout)

        submit_button.clicked.connect(self._on_submit)
        cancel_button.clicked.connect(self.reject)
        
    def _create_fields(self, fields):

        for field_name, config in fields.items():
            field_label = QLabel(config.get('label', field_name))
            self.layout.addWidget(field_label)

            field_type = config.get('type', 'QLineEdit')
            field_widget = self._create_widget(field_type, config)

            if field_widget:
                self.layout.addWidget(field_widget)
                self.input_fields[field_name] = field_widget 
                
    def _create_widget(self, widget_type, config):
        if widget_type == 'QLineEdit':
            widget = QLineEdit()
            widget.setPlaceholderText(config.get('placeholder', ''))
            widget.setText(config.get('default', ''))
            return widget
        elif widget_type == 'QComboBox':
            widget = QComboBox()
            widget.addItems(config.get('options', []))
            return widget
        elif widget_type == 'QCheckBox':
            widget = QCheckBox()
            widget.setChecked(config.get('default', False))
            return widget
        elif widget_type == 'QSpinBox':
            widget = QSpinBox()
            widget.setMinimum(config.get('range', [0, 100])[0])
            widget.setMaximum(config.get('range', [0, 100])[1])
            widget.setValue(config.get('default', 0) or 0)  # 防止 None 值
            return widget
        elif widget_type == 'QDoubleSpinBox':
            widget = QDoubleSpinBox()
            widget.setDecimals(2)
            widget.setMinimum(config.get('range', [0.0, 1.0])[0])
            widget.setMaximum(config.get('range', [0.0, 1.0])[1])
            widget.setValue(config.get('default', 0.0) or 0.0)  # 防止 None 值
            return widget
        return None

    def get_inputs(self):
        inputs = {}
        for field_name, widget in self.input_fields.items():
            if isinstance(widget, QLineEdit):
                inputs[field_name] = widget.text()
            elif isinstance(widget, QComboBox):
                inputs[field_name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                inputs[field_name] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                inputs[field_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                inputs[field_name] = widget.value()
        return inputs

    def _on_submit(self):
        missing_fields = [name for name, widget in self.input_fields.items() if isinstance(widget, QLineEdit) and not widget.text()]
        if missing_fields:
            QMessageBox.warning(self, 'warnings', f"The following fields are required :{', '.join(missing_fields)}")
        else:
            self.accept()

    def set_help_content(self, help_text):
        self.help_content = help_text

    def show_help_dialog(self):
        if self.help_content:
            QMessageBox.information(self, "Help", self.help_content)
        else:
            QMessageBox.information(self, "Help", "There is no help information available at this time.")
