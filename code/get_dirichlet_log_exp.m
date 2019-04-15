function e_log_pi = get_dirichlet_log_exp(alpha)

    e_log_pi = psi(alpha) - psi(sum(alpha));

return