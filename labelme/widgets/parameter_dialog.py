from PyQt5.QtWidgets import (
     QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QDoubleSpinBox, QAbstractSpinBox
)
from ..ai.esam.esam_everything import EfficientSAM_Everything

class ParameterDialog(QDialog):
    def __init__(self, esam_instance, parent=None):
        super().__init__(parent)
        self.esam_instance = esam_instance  
        self.init_ui()  

    def init_ui(self):
        self.setWindowTitle('Parameter Settings')

        layout = QFormLayout()

        self.param1 = QDoubleSpinBox(self)
        self.param1.setRange(0, 1.0)
       
        self.param2 = QSpinBox(self)
        self.param2.setRange(0, 400)

        self.param1.setButtonSymbols(QAbstractSpinBox.NoButtons)

        layout.addRow('NMS thresh:', self.param1)
        layout.addRow('Min Filter Area:', self.param2)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def getParameters(self):
        return self.esam_instance.getNMS(), self.esam_instance.getMinFilterArea()

    def setParameters(self):
        self.esam_instance.setNMS(self.param1.value())
        self.esam_instance.setMinFilterArea(self.param2.value())

    def load_parameters(self):
        nms_thresh, min_filter_area = self.getParameters()
        print(nms_thresh, min_filter_area)
        self.param1.setValue(nms_thresh)
        self.param2.setValue(min_filter_area)
