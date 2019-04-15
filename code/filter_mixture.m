function [w_mix, pi, hessian_mix, A_mix] = filter_mixture(w_mix_init, pi_init, hessian_mix_init, A_mix_init)
    idx_good = (pi_init >= 1e-10);
    w_mix = w_mix_init(:, idx_good);
    pi = pi_init(idx_good);
    pi = pi ./ sum(pi);
    hessian_mix = hessian_mix_init(idx_good, 1);
    A_mix = A_mix_init(idx_good, 1);
return