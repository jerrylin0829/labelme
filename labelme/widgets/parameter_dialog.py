from PyQt5.QtWidgets import (
     QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QDoubleSpinBox, QAbstractSpinBox,QComboBox
)
from ..ai.eSam.esam_everything import FILTER_MODE

class ParameterDialog(QDialog):
    def __init__(self, esam_instance, parent=None):
        super().__init__(parent)
        self.esam_instance = esam_instance  
        self.init_ui()  

    def init_ui(self):
        self.setWindowTitle('Parameter Settings')

        layout = QFormLayout()
        self.nms_thresh = QDoubleSpinBox(self)
        self.nms_thresh.setRange(0, 1.0)
        self.nms_thresh.setButtonSymbols(QAbstractSpinBox.NoButtons)
        
        self.min_filter_area = QSpinBox(self)
        self.min_filter_area.setRange(0, 400)
        self.min_filter_area.setButtonSymbols(QAbstractSpinBox.NoButtons)
        
        self.delta = QSpinBox(self)
        self.delta.setRange(1, 400)
        self.delta.setButtonSymbols(QAbstractSpinBox.NoButtons)
        
        self.IQR = QDoubleSpinBox(self)
        self.IQR.setRange(0, 3.0)
        self.IQR.setButtonSymbols(QAbstractSpinBox.NoButtons)
                
        self.filter_mode = QComboBox(self)
        self.filter_mode.addItems(FILTER_MODE)
        
        layout.addRow('NMS thresh:', self.nms_thresh)
        layout.addRow('Min Filter Area (MFA):', self.min_filter_area)
        layout.addRow('Component delta:', self.delta)
        layout.addRow('IQR:', self.IQR)
        layout.addRow('Filter Mode:', self.filter_mode)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def getParameters(self):
        return self.esam_instance.getNMS(), self.esam_instance.getMinFilterArea(), self.esam_instance.getDelta(),self.esam_instance.getIQR(),self.esam_instance.getFliterMode(),

    def setParameters(self):
        self.esam_instance.setNMS(self.nms_thresh.value())
        self.esam_instance.setMinFilterArea(self.min_filter_area.value())
        self.esam_instance.setDelta(self.delta.value())
        self.esam_instance.setIQR(self.IQR.value())
        self.esam_instance.setFliterMode(self.filter_mode.currentIndex())
        
    def load_parameters(self):
        nms_thresh, min_filter_area, delta, IQR, filter_mode= self.getParameters()
        self.nms_thresh.setValue(nms_thresh)
        self.min_filter_area.setValue(min_filter_area if min_filter_area is not None else 0)
        if filter_mode == 0 :
            self.IQR.setValue(IQR)
        else:
            self.delta.setValue(delta)
        self.filter_mode.setCurrentIndex(filter_mode)
