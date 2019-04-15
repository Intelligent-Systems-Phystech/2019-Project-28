function [A, w, pi, hessian, L] = maximize_evidence_mixture_logistic_laplace(X, y, K, alpha, num_start, A_0, w_0, eps, lambda, max_iter, grad_eps, em_max_iter, pi_eps)
    m = size(X, 1);
    n = size(X, 2);
    if (nargin < 13)
        pi_eps = 1e-10;
    end
    if (nargin < 12)
       em_max_iter = 40; 
    end
    if (nargin < 11)
       grad_eps = 1e-4; 
    end
    if (nargin < 10)
       max_iter = 20; 
    end
    if (nargin < 9)
       lambda = 1.; 
    end
    if (nargin < 8)
       eps = 1e-5; 
    end
    if (nargin < 7)
        w_0 = zeros(n, K);
    end
    if (nargin < 6)
        A_0 = cell(K, 1);
        for k=1:K
            A_0{k} = eye(n);
        end
    end
    if (nargin < 5)
       num_start = 20; 
    end
    
    L = nan * ones(em_max_iter + 1, num_start);
    pi_best = cell(num_start, 1);
    w_best = cell(num_start, 1);
      
    for index=1:num_start
    
        % Initialization
        if (rand(1) > 0.5)
            pi = rand(K, 1);
        else
            pi = gamrnd(0.1 * ones(K, 1), ones(K, 1)); % making slightly sparse dirichlet sampling
        end
        pi = pi / sum(pi);    
        w = randn(n, K);
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
                [w(:, k), hessian{k}] = learn_single_logistic(X, y, A{k}, eps, lambda, max_iter, grad_eps, prob(:, k)); 
               end
            end
            %L(i, 1) = count_mixture_learn_L(X, y, A, alpha, pi, w, prob);
            if (sum(abs(pi - pi_old)) < pi_eps)
               break; 
            end
            pi_old = pi;
            %pi
        end
        L(i + 1, index) = count_mixture_learn_L(X, y, A, alpha, pi, w, prob);

        pi_best{index} = pi;
        w_best{index} = w;
    end
    L_min = nanmax(L, [], 1);
    [~, idx_max] = max(L_min);
    w = w_best{idx_max};
    pi = pi_best{idx_max};
    
return