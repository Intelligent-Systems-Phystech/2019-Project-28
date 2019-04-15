function L = count_mixture_evidence_vlb_L(X, y, A, alpha, alphavec, vvec, tilde_A, prob, xi)
    K = size(A, 1);
    m = size(X, 1);
    n = size(X, 2);
    
    lambda_xi = get_lambda(xi);
    sigma = ones(m, K) ./ (1 + exp(-xi));
    log_pi_exp = get_dirichlet_log_exp(alphavec);
    
    L_pi = gammaln(K * alpha) - K * gammaln(alpha) + (alpha - 1) * sum(log_pi_exp);
    L_xi = sum(sum(prob .* log(1e-20 + sigma))) - 0.5 * sum(sum(prob .* xi));
    L_w = 0.5 * sum(sum(prob .* ((X .* (y * ones(1, n))) * vvec)));
    L_lambda = sum(sum(lambda_xi .* xi .* xi .* prob));
    L_quad = 0;
    for k=1:K
        cur_matr = vvec(:, k) * vvec(:, k)' + inv(tilde_A{k} + 1e-10);
        for i=1:m
            L_quad = L_quad - prob(i, k) * lambda_xi(i, k) * (X(i, :) * cur_matr * X(i, :)');
        end
    end
    L_a = 0;
    for k=1:K
        L_a = L_a + 0.5 * sum(log(diag(A{k}) + 1e-20));
        cur_matr = vvec(:, k) * vvec(:, k)' + inv(tilde_A{k} + 1e-10);
        L_a = L_a - 0.5 * sum(diag(A{k}) .* diag(cur_matr));
    end
    L_pi_z = sum(prob, 1) * log_pi_exp;
    
    L_q_z = sum(sum(prob .* log(prob + 1e-20)));
    L_q_w = 0;
    for k=1:K
       L_q_w = L_q_w + 0.5 * sum(log(diag(tilde_A{k} + 1e-10))); 
    end
   
    L_q_pi = gammaln(sum(alphavec)) - sum(gammaln(alphavec)) + sum((alphavec - 1) .* log_pi_exp);
    
    L = -(L_q_pi + L_q_z + L_q_w) + L_pi + L_xi + L_w + L_lambda + L_quad + L_a + L_pi_z;

return