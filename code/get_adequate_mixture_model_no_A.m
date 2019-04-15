function [w, pi, hessian] = get_adequate_mixture_model_no_A(X, y, K, signif_alpha, eps, lambda, max_iter, grad_eps, em_max_iter, pi_eps, num_start, pi0, w0)
    if (nargin < 11)
       num_start = 20; 
    end
    if (nargin < 10)
        pi_eps = 1e-5;
    end
    if (nargin < 9)
       em_max_iter = 100; 
    end
    if (nargin < 8)
       grad_eps = 1e-4; 
    end
    if (nargin < 7)
       max_iter = 20; 
    end
    if (nargin < 6)
       lambda = 1.; 
    end
    if (nargin < 5)
       eps = 1e-5; 
    end
    if (nargin < 4)
       signif_alpha = 0.05; 
    end
    if (nargin >= 13)
       num_start = 1; 
    end
    n = size(X, 2);
    A = cell(K, 1);
    for k=1:K
       A{k} = zeros(n, n); 
    end
    
    if (nargin >= 13)
        [w, pi, hessian] = learn_mixture_logistic(X, y, A, 1., eps, lambda, max_iter, grad_eps, em_max_iter, pi_eps, num_start, 0, pi0, w0);
    else
        [w, pi, hessian] = learn_mixture_logistic(X, y, A, 1., eps, lambda, max_iter, grad_eps, em_max_iter, pi_eps, num_start, 0);
    end

    t_matr = get_significance_matrix_no_intersect_mixture(w, hessian);
    %size(t_matr), K
    
    t_matr = t_matr - eye(K);
    [~, index] = max(t_matr(:));
    [k, l] = ind2sub(size(t_matr), index);
    
    if (t_matr(k, l) < signif_alpha)
       return 
    end
    
    sum_pi = pi(k) + pi(l);
    sum_w = (pi(k) * w(:, k) + pi(l) * w(:, l)) / (sum_pi + 1e-20);
    
    w(:, k) = sum_w;
    w(:, l) = [];
    pi(k, 1) = sum_pi;
    pi(l, :) = []; 
    
    [w, pi, hessian] = get_adequate_mixture_model_no_A(X, y, K - 1, signif_alpha, eps, lambda, max_iter, grad_eps, em_max_iter, pi_eps, num_start, pi, w);
return