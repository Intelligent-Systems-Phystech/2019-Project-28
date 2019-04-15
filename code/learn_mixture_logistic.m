function [w, pi, hessian, L, pi_evolution, w_evolution] = learn_mixture_logistic(X, y, A, alpha, eps, lambda, max_iter, grad_eps, em_max_iter, pi_eps, num_start, illustrate, pi_0, w_0)
    if (nargin < 12)
       illustrate = 0; 
    end
    %illustrate = 1;
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
       grad_eps = 1e-10; 
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
    if (nargin >= 14)
       num_start = 1; 
    end
    
    m = size(X, 1);
    n = size(X, 2);
    K = size(A, 1);
    
    L = nan * ones(em_max_iter + 1, num_start);
    pi_best = cell(num_start, 1);
    w_best = cell(num_start, 1);
    hessian_best = cell(num_start, 1);
    
    pi_evolution = nan;
    w_evolution = nan;
    if (illustrate == 1)
       pi_evolution = cell(em_max_iter + 1, num_start);
       w_evolution = cell(em_max_iter + 1, num_start);
    end
    
    for index=1:num_start
        disp 'START'
        index
        % Initialization
        if (nargin >= 13)
            pi = pi_0;
        else
            if (rand(1) > 0.5)
                pi = rand(K, 1);
            else
                pi = gamrnd(0.05 * ones(K, 1), ones(K, 1)); % making slightly sparse dirichlet sampling
            end
            pi = pi / sum(pi);    
        end
        
        if (nargin >= 14)
            w = w_0;
        else
            w = randn(n, K);
        end
        pi_old = pi;
        
        hessian = cell(K, 1);
        is_ok = ones(K, 1);

        for i=1:em_max_iter
            % E-step
            sigma = ones(m, K) ./ (1 + exp(-(X * w) .* (y * ones(1, K))));
            prob = sigma .* (ones(m, 1) * pi');
            sum_prob = prob * ones(K, 1);
            prob = prob ./ (sum_prob * ones(1, K));

            L(i, index) = count_mixture_learn_L(X, y, A, alpha, pi, w, prob);

            % M-step
            pi = alpha - 1 + sum(prob, 1)';
            is_ok(pi < 0) = 0;
            pi = max(pi, 0);
            pi = pi / sum(pi);

            for k=1:K
               if (is_ok(k))
                [w(:, k), hessian{k}] = learn_single_logistic(X, y, A{k}, eps, lambda, max_iter, grad_eps, prob(:, k), w(:, k)); 
               else
                   if (size(hessian{k}, 1) == 0)
                     hessian{k} = zeros(n, n);
                   end
               end
            end
            %L(i, 1) = count_mixture_learn_L(X, y, A, alpha, pi, w, prob);
            if (illustrate == 1)
               pi_evolution{i, index} = pi;
               w_evolution{i, index} = w;
            end
            if (sum(abs(pi - pi_old)) < pi_eps)
               break; 
            end
            pi_old = pi;
            %pi
        end
        L(i + 1, index) = count_mixture_learn_L(X, y, A, alpha, pi, w, prob);
        if (illustrate == 1)
           pi_evolution{i + 1, index} = pi;
           w_evolution{i + 1, index} = w;
        end
        pi_best{index} = pi;
        w_best{index} = w;
        hessian_best{index} = hessian;
    end
    L_min = nanmax(L, [], 1);
    [~, idx_max] = max(L_min);
    w = w_best{idx_max};
    pi = pi_best{idx_max};
    hessian = hessian_best{idx_max};
    
return