import unittest
import mujoco
import numpy as np
from .environment import Plant, Domain, sample_domain, reference, geometric, allocation, animate_propellers


class PhysicsTests(unittest.TestCase):
    def test_x_frame_geometry_and_motor_mixer(self):
        plant = Plant()
        offsets = np.array([plant.model.site(f"rotor{i}").pos[:2] for i in range(4)])
        np.testing.assert_allclose(np.abs(offsets), .24/np.sqrt(2), atol=1e-12)
        np.testing.assert_array_equal(np.sign(offsets), [[1, 1], [-1, 1], [-1, -1], [1, -1]])
        # An unobstructed centerline between the two front rotor discs.
        self.assertGreater(float(np.min(np.abs(offsets[:, 1]))), .118)
        for i in range(4):
            np.testing.assert_allclose(plant.model.geom(f"prop{i}").pos, plant.model.site(f"rotor{i}").pos)
        for torque in ([.1, 0, 0], [0, .1, 0], [0, 0, .05], [.08, -.09, .04]):
            plant.data.act[:] = allocation(12., torque, plant.trim)
            mujoco.mj_forward(plant.model, plant.data)
            np.testing.assert_allclose(plant.data.qfrc_actuator, [0, 0, 12., *torque], atol=1e-10)

    def test_x_frame_heading_tracking(self):
        for mode, kind in (("fixed", "figure8"), ("tangent", "figure8"), ("sweep", "hover")):
            plant = Plant(); plant.reset(reference(0, kind, heading=mode)[0])
            errors, attitude_errors = [], []
            for step in range(400):
                target, acc = reference(step*.05, kind, heading=mode)
                state, contact = plant.step(geometric(plant.state(), target, acc, plant.trim))
                desired = reference((step+1)*.05, kind, heading=mode)[0]
                self.assertFalse(contact)
                errors.append(np.linalg.norm(state[:3]-desired[:3]))
                delta = desired[6:15].reshape(3, 3).T @ state[6:15].reshape(3, 3)
                attitude_errors.append(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))
            self.assertLess(np.sqrt(np.mean(np.square(errors))), .10)
            self.assertLess(np.sqrt(np.mean(np.square(attitude_errors))), .10)

    def test_asset_body_axes_and_rotor_geometry(self):
        plant = Plant()
        # Check world transforms at two headings, not just the XML's local text.
        for yaw in (0., np.pi/2):
            plant.reset(reference(yaw/.45, "hover", heading="sweep")[0])
            R = plant.data.xmat[plant.body].reshape(3, 3)
            lens = plant.model.geom("camera_lens").id
            np.testing.assert_allclose(plant.data.geom_xmat[lens].reshape(3, 3)[:, 2], R[:, 0], atol=1e-7)
            for side in range(2):
                skid = plant.model.geom(f"skid{side}").id
                axis = plant.data.geom_xmat[skid].reshape(3, 3)[:, 2]
                self.assertAlmostEqual(abs(np.dot(axis, R[:, 0])), 1.)
            offsets = []
            for i in range(4):
                site = plant.model.site(f"rotor{i}").id
                prop = plant.model.geom(f"prop{i}").id
                np.testing.assert_allclose(plant.data.site_xpos[site], plant.data.geom_xpos[prop])
                np.testing.assert_allclose(plant.data.site_xmat[site].reshape(3, 3)[:, 2], R[:, 2], atol=1e-7)
                self.assertEqual(plant.model.geom_contype[prop], 0)
                self.assertEqual(plant.model.geom_conaffinity[prop], 0)
                offsets.append(plant.model.site_pos[site, :2])
            for i in range(4):
                self.assertAlmostEqual(float(np.dot(offsets[i], offsets[(i+1) % 4])), 0.)
                self.assertGreater(np.linalg.norm(offsets[i]-offsets[(i+1) % 4]), 2*.118)
        plant.reset()
        axes = [plant.data.geom_xmat[plant.model.geom(f"prop{i}").id].reshape(3, 3)[:, 0] for i in range(4)]
        self.assertAlmostEqual(float(np.dot(axes[0], axes[1])), 0.)

    def test_allocation_matches_mujoco_wrench(self):
        plant = Plant()
        desired = np.array([12., .08, -.09, .04])
        plant.data.act[:] = allocation(desired[0], desired[1:], plant.trim)
        mujoco.mj_forward(plant.model, plant.data)
        np.testing.assert_allclose(plant.data.qfrc_actuator, [0, 0, *desired], atol=1e-10)

    def test_visual_animation_does_not_change_flight(self):
        plain, animated = Plant(), Plant()
        for step in range(60):
            target, acc = reference(step*.05)
            action = geometric(plain.state(), target, acc, plain.trim)
            expected, expected_contact = plain.step(action)
            animate_propellers(animated, step*.05)
            actual, contact = animated.step(action)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(contact, expected_contact)

    def test_invalid_domain_rejected(self):
        for domain in (Domain(mass=-1), Domain(lag=0), Domain(drag=-.1)):
            with self.assertRaises(ValueError): Plant(domain)

    def test_trim_and_rotor_torque(self):
        plant = Plant(); initial = plant.state().copy()
        for _ in range(40): state, contact = plant.step(np.full(4, plant.trim))
        self.assertFalse(contact)
        np.testing.assert_allclose(state[:6], initial[:6], atol=1e-5)
        plant.reset(); action = np.full(4, plant.trim); action[1] += .5
        state, _ = plant.step(action)
        self.assertGreater(state[15], 0)
        self.assertLess(state[17], 0)

    def test_snapshot_restores_motor_state(self):
        plant = Plant(Domain(lag=.09)); plant.step(np.array([4., 3., 3., 3.]))
        snapshot = plant.snapshot(); expected, _ = plant.step(np.full(4, 3.2))
        plant.restore(snapshot); actual, _ = plant.step(np.full(4, 3.2))
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_shared_controller_tracks(self):
        plant = Plant(); plant.reset(reference(0)[0]); errors = []
        for step in range(240):
            target, acc = reference(step*.05)
            state, contact = plant.step(geometric(plant.state(), target, acc, plant.trim))
            self.assertFalse(contact)
            errors.append(np.linalg.norm(state[:3]-reference((step+1)*.05)[0][:3]))
        self.assertLess(np.sqrt(np.mean(np.square(errors))), .5)

    def test_combination_split(self):
        for seed in range(100):
            d = sample_domain(seed, "train"); self.assertFalse(d.mass > 1.40 and d.lag > .065)
            d = sample_domain(seed, "combination"); self.assertTrue(d.mass > 1.40 and d.lag > .065)

    def test_heading_tracking_and_wrap(self):
        for mode, kind in (("sweep", "hover"), ("tangent", "figure8")):
            plant = Plant(); plant.reset(reference(0, kind, heading=mode)[0])
            position_errors, attitude_errors = [], []
            for step in range(400):
                target, acc = reference(step*.05, kind, heading=mode)
                state, contact = plant.step(geometric(plant.state(), target, acc, plant.trim))
                desired = reference((step+1)*.05, kind, heading=mode)[0]
                self.assertFalse(contact)
                position_errors.append(np.linalg.norm(state[:3]-desired[:3]))
                delta = desired[6:15].reshape(3, 3).T @ state[6:15].reshape(3, 3)
                attitude_errors.append(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))
            self.assertLess(np.sqrt(np.mean(np.square(position_errors))), .25)
            self.assertLess(np.sqrt(np.mean(np.square(attitude_errors))), .20)


if __name__ == "__main__": unittest.main()
