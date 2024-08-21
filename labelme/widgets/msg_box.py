from PyQt5.QtWidgets import QMessageBox

class MessageBox:
    def __init__(self):
        pass

    def showMessageBox(self, title, message):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

