import unittest
import numpy as np
import torch
from .models import WorldModel, Readout, predict
from .train import batch


class ModelTests(unittest.TestCase):
    def test_capacity_gradient_and_basis(self):
        counts = []
        for variant in ("standard", "opf"):
            torch.manual_seed(7)
            model = WorldModel(variant, np.zeros(18), np.ones(18), np.zeros(4), np.ones(4)).cuda()
            counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
            states = torch.randn(4, 25, 18, device="cuda")
            actions = torch.randn(4, 24, 4, device="cuda")
            loss, _ = model.loss(states, actions); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            model.update_teacher()
            if variant == "opf": torch.testing.assert_close(model.basis@model.basis.T, torch.eye(24, device="cuda"))
            for kind in ("physics", "learned"):
                readout = Readout(kind).cuda()
                out = predict(model, readout, states[:, :10], actions[:, :9], actions[:, 9:])
                self.assertEqual(tuple(out.shape), (4, 15, 18)); self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(counts[0], counts[1])

    def test_window_sampling(self):
        data = {"states": torch.zeros(8, 201, 18, device="cuda"), "actions": torch.zeros(8, 200, 4, device="cuda")}
        states, actions = batch(data, np.random.default_rng(7), 16)
        self.assertEqual(tuple(states.shape), (16, 25, 18))
        self.assertEqual(tuple(actions.shape), (16, 24, 4))


if __name__ == "__main__": unittest.main()
