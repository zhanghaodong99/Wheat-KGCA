from .convnextv2_attention_variants import ConvNeXtV2AttentionVariant


def build_model(num_classes: int = 6, pretrained: bool = True):
    return ConvNeXtV2AttentionVariant(
        num_classes=num_classes,
        pretrained=pretrained,
        attention="res_eca2",
        head="linear",
    )
