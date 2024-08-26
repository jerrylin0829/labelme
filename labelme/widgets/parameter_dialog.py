from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QDoubleSpinBox, QAbstractSpinBox, QComboBox, QMessageBox
)
from labelme.logger import logger
from .qdialog_generator import CustomDialog


class ParameterDialog(CustomDialog):
    def __init__(self, esam_instance):
        fields = {
            'NMS thresh': {'type': 'QDoubleSpinBox', 'range': [0, 1.0], 'default': esam_instance.getNMS()},
            'Upper percentile': {'type': 'QSpinBox', 'range': [0, 100], 'default': esam_instance.getPctUp()},
            'Lower percentile': {'type': 'QSpinBox', 'range': [0, 100], 'default': esam_instance.getPctLow()},
            'Min Filter Area (MFA)': {'type': 'QSpinBox', 'range': [0, 400], 'default': esam_instance.getMinFilterArea()},
        }

        super().__init__(title="Parameter Settings", fields=fields,show_help=True)
        self.set_help_content = (
            "NMS thresh: 對於場景物件的敏感度 \n"
            "Upper percentile : 該比例以上之物體會被濾除 \n"
            "Lower percentile : 該比例以下之物體會被濾除\n"
            "Min Filter Area (MFA): Minimum area filter value (小於此面積會被濾除)\n"
        )
        self.esam_instance = esam_instance

    def setParameters(self):
        inputs = self.get_inputs()
        self.esam_instance.setNMS(inputs['NMS thresh'])
        self.esam_instance.setPctUp(inputs['Upper percentile'])
        self.esam_instance.setPctLow(inputs['Lower percentile'])
        self.esam_instance.setMinFilterArea(inputs['Min Filter Area (MFA)'])

