"""Matched EMA JEPA models and frozen-latent readouts for the MuJoCo study."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class WorldModel(nn.Module):
    def __init__(self, variant, state_mean, state_std, action_mean, action_std):
        super().__init__()
        self.variant = variant
        for key, value in (("sm", state_mean), ("ss", state_std), ("am", action_mean), ("ast", action_std)):
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(216, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 24))
        self.teacher = copy.deepcopy(self.encoder).requires_grad_(False)
        self.transition = nn.GRUCell(4, 24)
        self.head = nn.Linear(24, 24)
        # Equal parameter count: dense synthesis for standard; constrained
        # orthonormal factor synthesis for OPF. Both start at identity.
        self.basis = nn.Parameter(torch.eye(24))

    def encode(self, states, actions, teacher=False):
        x = torch.cat((((states-self.sm)/self.ss).flatten(-2),
                       ((actions-self.am)/self.ast).flatten(-2)), -1)
        return (self.teacher if teacher else self.encoder)(x)

    def rollout(self, states, history_actions, actions, mute=None):
        z = self.encode(states, history_actions)
        predictions, factors = [], []
        for a in ((actions-self.am)/self.ast).unbind(1):
            f = self.head(self.transition(a, z))
            if mute is not None:
                mask = torch.ones_like(f); mask[:, mute*6:(mute+1)*6] = 0
                f = f*mask
            z = f @ self.basis
            predictions.append(z); factors.append(f)
        return torch.stack(predictions, 1), torch.stack(factors, 1)

    @torch.no_grad()
    def update_teacher(self):
        for target, source in zip(self.teacher.parameters(), self.encoder.parameters()):
            target.lerp_(source, .01)
        if self.variant == "opf":
            q, r = torch.linalg.qr(self.basis.T)
            q = q * torch.where(torch.diag(r) < 0, -1., 1.)
            self.basis.copy_(q.T)

    def loss(self, states, actions):
        horizon = states.shape[1]-10
        z, factors = self.rollout(states[:, :10], actions[:, :9], actions[:, 9:9+horizon])
        with torch.no_grad():
            sw = states.unfold(1, 10, 1)[:, 1:].transpose(-1, -2).reshape(-1, 10, 18)
            aw = actions.unfold(1, 9, 1)[:, 1:1+horizon].transpose(-1, -2).reshape(-1, 9, 4)
            targets = self.encode(sw, aw, True).reshape(-1, horizon, 24)
        # OPF's target factorization and synthesis use the same basis.
        target_coordinates = targets @ self.basis.T if self.variant == "opf" else targets
        prediction = F.mse_loss(factors if self.variant == "opf" else z, target_coordinates)
        activity = F.relu(.3-torch.sqrt(target_coordinates.flatten(0, 1).var(0, unbiased=False)+1e-4)).mean()
        context = self.encode(states[:, :10], actions[:, :9])
        variance = F.relu(.3-torch.sqrt(context.var(0, unbiased=False)+1e-4)).mean()
        return prediction + .1*activity + .1*variance, prediction


def so3(v):
    x, y, z = v.unbind(-1); zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    theta = torch.linalg.vector_norm(v, dim=-1)
    a = torch.sinc(theta/torch.pi)[:, None, None]
    b = (.5*torch.sinc(theta/(2*torch.pi)).square())[:, None, None]
    return torch.eye(3, device=v.device) + a*skew + b*(skew@skew)


class Readout(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.net = nn.Sequential(nn.Linear(24, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 15 if kind == "physics" else 18))
        if kind == "physics":
            nn.init.normal_(self.net[-1].weight, std=.001)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, initial, actions, sm, ss):
        raw = self.net(z)
        if self.kind == "learned":
            # Direct state prediction: no thrust, gravity, SO(3) or integration
            # equations occur in this learned readout.
            return raw*ss+sm
        p, v, R, w = initial[:, :3], initial[:, 3:6], initial[:, 6:15].reshape(-1, 3, 3), initial[:, 15:18]
        predictions = []
        gravity = torch.tensor([0., 0., 9.81], device=z.device)
        for step in range(z.shape[1]):
            a = R[:, :, 2]*(actions[:, step].sum(-1, keepdim=True)/1.3)-gravity+raw[:, step, :3]
            alpha = (raw[:, step, 3:].reshape(-1, 3, 4) @ actions[:, step, :, None]).squeeze(-1)
            p, v, R, w = p+.05*v, v+.05*a, R@so3(.05*w), w+.05*alpha
            predictions.append(torch.cat((p, v, R.flatten(1), w), -1))
        return torch.stack(predictions, 1)


def predict(model, readout, states, history_actions, actions):
    z, _ = model.rollout(states, history_actions, actions)
    return readout(z, states[:, -1], actions, model.sm, model.ss)


def costs(predicted, targets, actions, reference_actions):
    weight = torch.tensor([400.]*3+[40.]*3+[20.]*12, device=predicted.device)
    return (((predicted-targets).square()*weight).sum(-1)+.05*(actions-reference_actions).square().sum(-1)).mean(-1)
