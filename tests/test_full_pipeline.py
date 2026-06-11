from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.models.ik import PoseToJointIK
from s2p2e.models.policy_bridge import PPEPolicyBridge


def test_ik_and_policy_bridge_shapes() -> None:
    import torch

    ik = PoseToJointIK()
    out = ik(torch.randn(2, 8))
    assert out["q_ref"].shape == (2, 7)

    bridge = PPEPolicyBridge(obs_dim=21, action_dim=7, ppe_feature_dim=512)
    action = bridge.act(torch.randn(2, 21), torch.randn(2, 512))
    assert action.shape == (2, 7)
