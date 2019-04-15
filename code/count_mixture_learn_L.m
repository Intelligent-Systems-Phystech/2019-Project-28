function L = count_mixture_learn_L(X, y, A, alpha, pi, w, prob)
    m = size(X, 1);
    K = size(w, 2);
    log_pi = log(pi + 1e-10);
    L_pi = (sum(prob, 1) + alpha - 1) * log_pi;
    L_w = 0;
    for k=1:K
       L_w = L_w - 0.5 * w(:, k)' * A{k} * w(:, k); 
    end
    sigma = ones(m, K) ./ (1 + exp(-(X * w) .* (y * ones(1, K))));
    L_x = sum(sum(log(sigma + 1e-10) .* prob));
    
    L_z = -sum(sum(prob .* log(prob + 1e-10)));
    L = L_x + L_w + L_pi + L_z;
    
return