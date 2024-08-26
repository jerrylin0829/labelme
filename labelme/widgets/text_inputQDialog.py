from PyQt5.QtWidgets import QDialog
from labelme.logger import logger
from .qdialog_generator import CustomDialog

class StringInputDialog(CustomDialog):
    def __init__(self, title="Title", default_text=""):
        fields = {
            'User Input': {
                'type': 'QLineEdit',
                'placeholder': 'Please enter the content',
                'default': default_text
            }
        }
        super().__init__(title=title, fields=fields)

    def get_user_input(self):
        return self.get_inputs().get('User Input', None)

def create_string_input_dialog(title="Title", default_text=""):
    dialog = StringInputDialog(title=title, default_text=default_text)
    result = dialog.exec_()
    
    if result == QDialog.Accepted:
        user_input = dialog.get_user_input()
        logger.info(f"User input: {user_input}")
        return user_input
    else:
        logger.info("Dialog was cancelled.")
        return None
