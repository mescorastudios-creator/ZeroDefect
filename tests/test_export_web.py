import numpy as np
import torch
import torch.nn.functional as F

from backend.export_web import export_backbone
from inspection.patchcore import Backbone


def test_folded_weights_reproduce_conv_and_batch_norm():
    class Stub:
        backbone = Backbone()

    data, spec = export_backbone(Stub)
    flat = np.frombuffer(data, np.float16).astype(np.float32)
    s = spec["layer1.0.conv1"]
    w = torch.from_numpy(flat[s["w"] : s["b"]].reshape(s["shape"])).permute(3, 2, 0, 1)  # HWIO -> OIHW
    b = torch.from_numpy(flat[s["b"] : s["b"] + s["shape"][3]])
    block = Stub.backbone.stem[4][0]
    x = torch.randn(1, 64, 20, 20)
    with torch.no_grad():
        expected = block.bn1(block.conv1(x))
        got = F.conv2d(x, w, b, padding=1)
    assert torch.allclose(got, expected, atol=0.05)
