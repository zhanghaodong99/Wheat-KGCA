from .base_timm_wrapper import TimmBackboneWrapper


class MobileNetV4HybridMedium224(TimmBackboneWrapper):
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__(
            model_name="mobilenetv4_hybrid_m",
            timm_model_name="mobilenetv4_hybrid_medium",
            num_classes=num_classes,
            pretrained=pretrained,
            input_size=224,
        )


def build_model(num_classes: int = 6, pretrained: bool = True):
    return MobileNetV4HybridMedium224(num_classes=num_classes, pretrained=pretrained)
