import sys
from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout

# 自定義對話框類
class CustomDialog(QDialog):
    def __init__(self, title="Custom Dialog", fields=None, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(title)
        
        self.input_fields = {} 

        layout = QVBoxLayout()

        if fields is not None:
            for field in fields:
                field_label = QLabel(field)
                field_input = QLineEdit()
                layout.addWidget(field_label)
                layout.addWidget(field_input)
                self.input_fields[field] = field_input  
                
        button_layout = QHBoxLayout()
        submit_button = QPushButton("Submit")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(submit_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        submit_button.clicked.connect(self.accept)  
        cancel_button.clicked.connect(self.reject)  
        
    def get_inputs(self):
        inputs = {field: self.input_fields[field].text() for field in self.input_fields}
        print("Inputs:", inputs)  
        return inputs


def create_customBox(fields):
    dialog = CustomDialog(fields=fields)
    result = dialog.exec_() 
    
    if result == QDialog.Accepted:
        inputs = dialog.get_inputs()  
        print("User inputs:", inputs)  
        
        def get_attribute(self, attribute_name):
            try:
                return getattr(self, attribute_name)
            except AttributeError:
                available_attributes = ', '.join(self.__dict__.keys())
                raise AttributeError(f"No such attribute: {attribute_name}. Available attributes: {available_attributes}")
        
        user_instance = type('DynamicUserInstance', (object,), {**inputs, 'get_attribute': get_attribute})()

        print("User instance created successfully with fields:", fields)
        return user_instance
    else:
        print("Dialog was cancelled.")
        return None

