from labelme.logger import logger
from .qdialog_generator import CustomDialog


class ParameterDialog(CustomDialog):
    def __init__(self, esam_instance):
        fields = {
            'Upper percentile': {'type': 'QSpinBox', 'range': [0, 100], 'default': esam_instance.get_parameters("PctUp")},
            'Lower percentile': {'type': 'QSpinBox', 'range': [0, 100], 'default': esam_instance.get_parameters("PctLow")},
            'Min Filter Area (MFA)': {'type': 'QSpinBox', 'range': [0, 400], 'default': esam_instance.get_parameters("MinFilterArea")},
        }

        super().__init__(title="Parameter Settings", fields=fields,show_help=True)
        self.set_help_content = (
            "Upper percentile : 該比例以上之物體會被濾除 \n"
            "Lower percentile : 該比例以下之物體會被濾除\n"
            "Min Filter Area (MFA): Minimum area filter value (小於此面積會被濾除)\n"
        )
        self.esam_instance = esam_instance

    def setParameters(self):
        inputs = self.get_inputs()
        self.esam_instance.set_parameters(set_type = "PctUp", val = inputs['Upper percentile'])
        self.esam_instance.set_parameters(set_type = "PctLow", val = inputs['Lower percentile'])
        self.esam_instance.set_parameters(set_type = "MinFilterArea", mfa = inputs['Min Filter Area (MFA)'])

