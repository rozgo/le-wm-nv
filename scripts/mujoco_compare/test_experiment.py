import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .environment import reference, prior
from .experiment import control_tasks, trace_name, validate_data
from .reference_plan import ReferencePlan


class ContractTests(unittest.TestCase):
    def test_cached_prior_matches_direct_calculation(self):
        for heading in ("fixed", "tangent", "sweep"):
            plan = ReferencePlan(240, 3.4, "figure8", heading)
            for step in (0, 1, 3, 19, 237):
                state = reference(step*.05, "figure8", heading=heading)[0]
                state[:3] += [.1, -.05, .02]
                actions, targets = prior(state, step*.05, 3.4, kind="figure8", heading=heading)
                actual_actions, actual_targets = plan.at(state, step)
                np.testing.assert_allclose(actual_actions, actions, rtol=0, atol=1e-6)
                np.testing.assert_allclose(actual_targets, targets, rtol=0, atol=1e-6)

    def test_airframe_and_heading_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"manifest.json").write_text(json.dumps({"airframe": "x", "heading_modes": ["fixed", "tangent", "sweep"]}))
            with self.assertRaises(ValueError): validate_data(root, {"airframe": "legacy"})
            with self.assertRaises(ValueError): validate_data(root, {"airframe": "x", "heading_modes": ["fixed"]})
            validate_data(root, {"airframe": "x", "heading_modes": ["fixed", "tangent", "sweep"]})

    def test_heading_traces_cannot_overwrite_one_another(self):
        names = {trace_name("opf-learned-7", "test", 0, "figure8", h, 2) for h in ("fixed", "tangent", "sweep")}
        self.assertEqual(len(names), 3)
        self.assertEqual(trace_name("standard-physics-7", "combination", 0, "figure8"), "standard-physics-7-combination-0-figure8.json")
        self.assertEqual(control_tasks({"control": {"kinds": ["circle"]}}), [{"kind": "circle", "heading": "fixed"}])


if __name__ == "__main__": unittest.main()
