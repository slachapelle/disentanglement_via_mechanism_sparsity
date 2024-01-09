import math
import torch


class GumbelSigmoid(torch.nn.Module):
    def __init__(self, shape, freeze=False, drawhard=True, tau=1, one_gumbel_sample=False):
        super(GumbelSigmoid, self).__init__()
        self.shape = shape
        self.freeze = freeze
        self.drawhard = drawhard
        self.log_alpha = torch.nn.Parameter(torch.zeros(self.shape))
        self.tau = tau
        self.one_sample_per_batch = one_gumbel_sample
        # useful to make sure these parameters will be pushed to the GPU
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.register_buffer("fixed_mask", torch.ones(shape))
        self.reset_parameters()

    def forward(self, bs):
        if self.freeze:
            y = self.fixed_mask.unsqueeze(0).expand((bs,) + self.shape)
            return y
        else:
            shape = tuple([bs] + list(self.shape))
            logistic_noise = self.sample_logistic(shape).type(self.log_alpha.type()).to(self.log_alpha.device)
            y_soft = torch.sigmoid((self.log_alpha + logistic_noise) / self.tau)

            if self.drawhard:
                y_hard = (y_soft > 0.5).type(y_soft.type())

                # This weird line does two things:
                #   1) at forward, we get a hard sample.
                #   2) at backward, we differentiate the gumbel sigmoid
                y = y_hard.detach() - y_soft.detach() + y_soft

            else:
                y = y_soft

            return y

    def get_proba(self):
        """Returns probability of getting one"""
        if self.freeze:
            return self.fixed_mask
        else:
            return torch.sigmoid(self.log_alpha)

    def reset_parameters(self):
        torch.nn.init.constant_(self.log_alpha, 5) # 5)  # will yield a probability ~0.99. Inspired by DCDI

    def sample_logistic(self, shape):
        if self.one_sample_per_batch:
            bs = shape[0]
            u = self.uniform.sample([1] + list(self.shape)).expand((bs,) + self.shape)
            return torch.log(u) - torch.log(1 - u)
        else:
            u = self.uniform.sample(shape)
            return torch.log(u) - torch.log(1 - u)

    def threshold(self):
        proba = self.get_proba()
        self.fixed_mask.copy_((proba > 0.5).type(proba.type()))
        self.freeze = True


class LouizosGumbelSigmoid(torch.nn.Module):
    """My implementation of https://openreview.net/pdf?id=H1Y8hhg0b"""
    def __init__(self, shape, freeze=False, tau=1, gamma=-0.1, zeta=1.1):
        super(LouizosGumbelSigmoid, self).__init__()
        self.shape = shape
        self.freeze=freeze
        self.log_alpha = torch.nn.Parameter(torch.zeros(self.shape))
        self.tau = tau
        assert gamma < 0 and zeta > 1
        self.gamma = gamma
        self.zeta = zeta
        # useful to make sure these parameters will be pushed to the GPU
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.register_buffer("fixed_mask", torch.ones(shape))
        self.reset_parameters()

    def forward(self, bs):
        if self.freeze:
            y = self.fixed_mask.unsqueeze(0).expand((bs,) + self.shape)
            return y
        else:
            shape = tuple([bs] + list(self.shape))
            logistic_noise = self.sample_logistic(shape).type(self.log_alpha.type()).to(self.log_alpha.device)
            y_soft = torch.sigmoid((self.log_alpha + logistic_noise) / self.tau)
            y_soft = y_soft * (self.zeta - self.gamma) + self.gamma
            one = torch.ones((1,)).to(self.log_alpha.device)
            zero = torch.zeros((1,)).to(self.log_alpha.device)
            y_soft = torch.minimum(one , torch.maximum(zero, y_soft))

            return y_soft

    def get_proba(self):
        """Returns probability of mask being > 0"""
        if self.freeze:
            return self.fixed_mask
        else:
            return torch.sigmoid(self.log_alpha - self.tau * (math.log(-self.gamma) - math.log(self.zeta)))

    def reset_parameters(self):
        #torch.nn.init.constant_(self.log_alpha, 5) # 5)  # will yield a probability ~0.99. Inspired by DCDI
        torch.nn.init.constant_(self.log_alpha, self.tau * (math.log(1 - self.gamma) - math.log(self.zeta - 1)))  # at init, half the samples will be exactly one.
        # general formula is p(M = 1) = sigmoid(log_alpha - beta (log(1-gamma) - log(zeta - 1))
        print(f"initialized so that P(mask != 0) = {self.get_proba().view(-1)[0]}")

    def sample_logistic(self, shape):
        u = self.uniform.sample(shape)
        return torch.log(u) - torch.log(1 - u)

    def threshold(self):
        proba = self.get_proba()
        self.fixed_mask.copy_((proba > 0.5).type(proba.type()))
        self.freeze = True
