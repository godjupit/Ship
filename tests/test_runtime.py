import unittest
import builtins
import os
from pathlib import Path
from unittest.mock import patch

import torch

from supership.generate_submission import build_runtime


class RuntimeTests(unittest.TestCase):
    def test_position_encoding_moves_without_changing_checkpoint(self):
        original_open = builtins.open
        data_root = Path("data").resolve()

        def guarded_open(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                resolved = Path(os.fsdecode(path)).resolve()
                self.assertFalse(resolved.is_relative_to(data_root), f"init reads data: {resolved}")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=guarded_open):
            _, model, _ = build_runtime(
                "configs/dfine/dfine_hgnetv2_s_supership.yml", "weights/baseline.pth", "cpu"
            )
        encoder = model.encoder
        for index in encoder.use_encoder_idx:
            name = f"pos_embed{index}"
            self.assertIn(name, dict(encoder.named_buffers()))
            self.assertNotIn(name, encoder.state_dict())
        model.to(dtype=torch.float64)
        for index in encoder.use_encoder_idx:
            self.assertEqual(getattr(encoder, f"pos_embed{index}").dtype, torch.float64)


if __name__ == '__main__':
    unittest.main()
