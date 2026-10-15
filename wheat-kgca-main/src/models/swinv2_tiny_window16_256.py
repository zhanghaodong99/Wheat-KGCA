from .base_timm_wrapper import TimmBackboneWrapper


class SwinV2TinyWindow16_256(TimmBackboneWrapper):
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__(
            model_name="swinv2_tiny_window16_256",
            timm_model_name="swinv2_tiny_window16_256",
            num_classes=num_classes,
            pretrained=pretrained,
            input_size=256,
        )


def build_model(num_classes: int = 6, pretrained: bool = True):
    return SwinV2TinyWindow16_256(num_classes=num_classes, pretrained=pretrained)
