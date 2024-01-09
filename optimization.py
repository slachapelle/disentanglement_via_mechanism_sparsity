import torch
import cooper


def compute_nll(log_likelihood, valid, opt):
    valid = valid.type(log_likelihood.type())
    if opt.include_invalid:
        nll = - torch.mean(log_likelihood)
    else:
        nll = - torch.dot(log_likelihood, valid) / torch.sum(valid)

    return nll


class CustomCMP(cooper.ConstrainedMinimizationProblem):
    def __init__(self, g_reg_coeff=0., gc_reg_coeff=0., g_constraint=0, gc_constraint=0., g_scaling=1., gc_scaling=1.,
                 schedule=False, max_g=0, max_gc=0):
        self.is_constrained = (g_constraint > 0. or gc_constraint > 0.)
        self.g_reg_coeff = g_reg_coeff
        self.gc_reg_coeff = gc_reg_coeff
        self.g_constraint = g_constraint
        self.gc_constraint = gc_constraint

        if schedule:
            self.current_g_constraint = max_g
            self.current_gc_constraint = max_gc
        else:
            self.current_g_constraint = g_constraint
            self.current_gc_constraint = gc_constraint

        self.max_g, self.max_gc = max_g, max_gc

        self.g_scaling = g_scaling
        self.gc_scaling = gc_scaling
        super().__init__(is_constrained=self.is_constrained)

    def closure(self, model, obs, cont_c, disc_c, valid, other, opt):
        misc = {}
        if opt.mode == "vae":
            elbo, reconstruction_loss, kl, _ = model.elbo(obs, cont_c, disc_c)
            loss = compute_nll(elbo, valid, opt)
        elif opt.mode == "supervised_vae":
            obs = obs[:, -1]
            other = other[:, -1]
            z_hat = model.latent_model.mean(model.latent_model.transform_q_params(model.encode(obs)))
            loss = torch.mean((z_hat.view(z_hat.shape[0], -1) - other) ** 2)
            reconstruction_loss, kl = 0, 0
        elif opt.mode == "latent_transition_only":
            ll = model.log_likelihood(other, cont_c, disc_c)
            loss = compute_nll(ll, valid, opt)
            reconstruction_loss, kl = 0, 0
        else:
            raise NotImplementedError(f"--mode {opt.mode} is not implemented.")

        misc["nll"] = loss.item()
        misc["reconstruction_loss"] = reconstruction_loss
        misc["kl"] = kl

        # regularization/constraint
        g_reg = model.latent_model.g_regularizer()
        gc_reg = model.latent_model.gc_regularizer()

        misc["g_reg"], misc["gc_reg"] = g_reg.item(), gc_reg.item()

        if not self.is_constrained:
            if self.g_reg_coeff > 0:
                loss += opt.g_reg_coeff * g_reg * self.g_scaling
            if self.gc_reg_coeff > 0:
                loss += opt.gc_reg_coeff * gc_reg * self.gc_scaling

            return cooper.CMPState(loss=loss, ineq_defect=None, eq_defect=None, misc=misc)
        else:
            defects = []
            if self.g_constraint > 0:
                defects.append(g_reg - self.current_g_constraint)
            if self.gc_constraint > 0:
                defects.append(gc_reg - self.current_gc_constraint)

            defects = torch.stack(defects)

            return cooper.CMPState(loss=loss, ineq_defect=defects, eq_defect=None, misc=misc)

    def update_constraint(self, iter, total_iter, no_update_period=0):
        if iter <= no_update_period:
            if self.g_constraint > 0:
                self.current_g_constraint = self.max_g
            if self.gc_constraint > 0:
                self.current_gc_constraint = self.max_gc
        elif no_update_period < iter <= no_update_period + total_iter:
            if self.g_constraint > 0:
                self.current_g_constraint = (self.max_g - self.g_constraint) * (1 - iter / total_iter) + self.g_constraint
            if self.gc_constraint > 0:
                self.current_gc_constraint = (self.max_gc - self.gc_constraint) * (1 - iter / total_iter) + self.gc_constraint
        else:
            if self.g_constraint > 0:
                self.current_g_constraint = self.g_constraint
            if self.gc_constraint > 0:
                self.current_gc_constraint = self.gc_constraint

        return self.current_g_constraint, self.current_gc_constraint

    def update_constraint_adaptive(self, iter, decrease_rate=0.0005, no_update_period=0):
        if iter <= no_update_period:
            if self.g_constraint > 0:
                self.current_g_constraint = self.max_g
            if self.gc_constraint > 0:
                self.current_gc_constraint = self.max_gc
        else:
            # decrease constraint only when defect is smaller than 0.1, otherwise do not change constraint.
            if self.g_constraint > 0 and self.state.ineq_defect.sum() <= 0.1:
                self.current_g_constraint = max(self.current_g_constraint - decrease_rate, self.g_constraint)
            if self.gc_constraint > 0 and self.state.ineq_defect.sum() <= 0.1:
                self.current_gc_constraint = max(self.current_gc_constraint - decrease_rate, self.gc_constraint)

        return self.current_g_constraint, self.current_gc_constraint

            #if self.g_constraint > 0 and self.state.ineq_defect.sum() <= 0:
            #    self.current_g_constraint = max(self.current_g_constraint - 1, self.g_constraint)
            #if self.gc_constraint > 0 and self.state.ineq_defect.sum() <= 0:
            #    self.current_gc_constraint = max(self.current_gc_constraint - 1, self.gc_constraint)

