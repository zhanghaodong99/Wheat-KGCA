from .base_timm_wrapper import TimmBackboneWrapper


class RepVITTiny224(TimmBackboneWrapper):
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__(
            model_name="repvit_tiny",
            timm_model_name="repvit_m0_9",
            num_classes=num_classes,
            pretrained=pretrained,
            input_size=224,
        )


def build_model(num_classes: int = 6, pretrained: bool = True):
    return RepVITTiny224(num_classes=num_classes, pretrained=pretrained)
