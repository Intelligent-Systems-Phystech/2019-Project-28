function [w, pi, hessian, A] = get_adequate_mixture_model(X, y, K, alpha, signif_alpha, pi_0, w_0, A_0, hessian_0, eps, lambda, max_iter, grad_eps, em_max_iter, pi_eps, num_start, a_eps)
    slight_opt = 0;
    if (nargin < 17)
       a_eps = 1e-5; 
    end
    if (nargin < 16)
       num_start = 10; 
    end
    if (nargin < 15)
        pi_eps = 1e-5;
    end
    if (nargin < 14)
       em_max_iter = 10; 
    end
    if (nargin < 13)
       grad_eps = 1e-4; 
    end
    if (nargin < 12)
       max_iter = 10; 
    end
    if (nargin < 11)
       lambda = 1.; 
    end
    if (nargin < 10)
       eps = 1e-5; 
    end
    if (nargin < 5)
       signif_alpha = 0.05; 
    end
    if (nargin < 4)
       alpha = 1.; 
    end
    if (nargin >= 6)
       num_start = 1;
       slight_opt = 1;
    end
    
    if (slight_opt == 1)
        w = w_0;
        pi = pi_0;
        A = A_0;
        hessian = hessian_0;
        %[w, pi, hessian, A, ~] = learn_optimal_mixture_logistic(X, y, K, alpha, eps, a_eps, em_max_iter, pi_eps, num_start, lambda, max_iter, grad_eps, pi_0, A_0, w_0);
    else
        [w, pi, hessian, A, ~] = learn_optimal_mixture_logistic(X, y, K, alpha, eps, a_eps, em_max_iter, pi_eps, num_start, lambda, max_iter, grad_eps);
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
    sum_A = max(A{k}, A{l});
    
    w(:, k) = sum_w;
    w(:, l) = [];
    pi(k, 1) = sum_pi;
    pi(l, :) = []; 
    A{k} = sum_A;
    A(l, :) = [];
    
    if (slight_opt == 1)
        [w, pi, hessian, A, ~] = learn_optimal_mixture_logistic(X, y, K - 1, alpha, eps, a_eps, em_max_iter, pi_eps, num_start, lambda, max_iter, grad_eps, pi, A, w);
    end
    
    [w, pi, hessian, A] = get_adequate_mixture_model(X, y, K - 1, alpha, signif_alpha, pi, w, A, hessian, eps, lambda, max_iter, grad_eps, em_max_iter, pi_eps, num_start, a_eps);
return