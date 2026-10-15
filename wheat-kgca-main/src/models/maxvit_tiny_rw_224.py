from .base_timm_wrapper import TimmBackboneWrapper


class MaxViTTinyRW224(TimmBackboneWrapper):
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__(
            model_name="maxvit_tiny_rw_224",
            timm_model_name="maxvit_tiny_rw_224",
            num_classes=num_classes,
            pretrained=pretrained,
            input_size=224,
        )


def build_model(num_classes: int = 6, pretrained: bool = True):
    return MaxViTTinyRW224(num_classes=num_classes, pretrained=pretrained)
