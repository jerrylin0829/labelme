from PyQt5.QtWidgets import (
     QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QDoubleSpinBox, QAbstractSpinBox,QComboBox
)
from ..ai.esam.esam_everything import EfficientSAM_Everything,FILTER_MODE

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
        self.param1.setButtonSymbols(QAbstractSpinBox.NoButtons)
        
        self.param2 = QSpinBox(self)
        self.param2.setRange(0, 400)
        self.param2.setButtonSymbols(QAbstractSpinBox.NoButtons)
        
        self.param3 = QSpinBox(self)
        self.param3.setRange(1, 400)
        self.param3.setButtonSymbols(QAbstractSpinBox.NoButtons)
        
        self.param4 = QDoubleSpinBox(self)
        self.param4.setRange(0, 3.0)
        self.param4.setButtonSymbols(QAbstractSpinBox.NoButtons)
                
        self.param5 = QComboBox(self)
        self.param5.addItems(FILTER_MODE)
        
        layout.addRow('NMS thresh:', self.param1)
        layout.addRow('Min Filter Area (MFA):', self.param2)
        layout.addRow('Component delta:', self.param3)
        layout.addRow('IQR:', self.param4)
        layout.addRow('Filter Mode:', self.param5)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def getParameters(self):
        return self.esam_instance.getNMS(), self.esam_instance.getMinFilterArea(), self.esam_instance.getDelta(),self.esam_instance.getIQR(),self.esam_instance.getFliterMode(),

    def setParameters(self):
        self.esam_instance.setNMS(self.param1.value())
        self.esam_instance.setMinFilterArea(self.param2.value())
        self.esam_instance.setDelta(self.param3.value())
        self.esam_instance.setIQR(self.param4.value())
        self.esam_instance.setFliterMode(self.param5.currentIndex())
        
    def load_parameters(self):
        nms_thresh, min_filter_area, delta, IQR, filter_mode= self.getParameters()
        self.param1.setValue(nms_thresh)
        self.param2.setValue(min_filter_area if min_filter_area is not None else 0)
        if filter_mode == 0 :
            self.param4.setValue(IQR)
        else:
            self.param3.setValue(delta)
        self.param5.setCurrentIndex(filter_mode)
