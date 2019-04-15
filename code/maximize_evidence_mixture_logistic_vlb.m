function [A, L, alphavec, vvec, tilde_A, prob, xi] = maximize_evidence_mixture_logistic_vlb(X, y, K, alpha, A_0, a_eps, eps, em_max_iter, num_start)
    m = size(X, 1);
    n = size(X, 2);
    if (nargin < 9)
       num_start = 5; 
    end
    if (nargin < 8)
       em_max_iter = 10; 
    end
    if (nargin < 7)
       eps = 1e-5; 
    end
    if (nargin < 6)
       a_eps = 1e-5; 
    end
        
    L = nan * ones(em_max_iter + 1, num_start);
    A_best = cell(num_start, 1);
    alphavec_best  = cell(num_start, 1);
    vvec_best  = cell(num_start, 1);
    tilde_A_best  = cell(num_start, 1);
    prob_best  = cell(num_start, 1);
    xi_best  = cell(num_start, 1);
         
    for index=1:num_start
    
        % Initialization
        alphavec = alpha * ones(K, 1) + 0.95 * alpha * (2 * rand(K, 1) - 1); % dirichlet parameters
        vvec = 5 * randn(n, K);
        if (nargin < 5)
           A_0 = cell(K, 1);
           for k=1:K
              A_0{k} = diag(10 * abs(randn(n, 1)));
           end
        end
        A = A_0;
        tilde_A = A_0;
        xi = 0.01 + 10 * abs(randn(m, K));
        %xi = ones(m, K);
        A_old = A;
        xi_old = xi;

        for iter=1:em_max_iter
            % E-step
            log_pi_exp = ones(m, 1) * get_dirichlet_log_exp(alphavec)';
            log_sigma = log(ones(m, K) ./ (1 + exp(-xi)));
            lambda_xi = get_lambda(xi);
            %size(X), size(mvec)
            prob = 0.5 * ((X * vvec) .* (y * ones(1, K))) - 0.5 * xi + lambda_xi .* xi .* xi;
            
            w_part = zeros(m, K);
            exp_wwt = cell(K, 1);
            for k=1:K
               exp_wwt{k} = vvec(:, k) * vvec(:, k)' + inv(tilde_A{k} + 1e-10); 
               for i=1:m
                  w_part(i, k) = X(i, :) * exp_wwt{k} * X(i, :)'; 
               end
            end
            w_part = -w_part .* lambda_xi;
            prob = prob + log_pi_exp + log_sigma + w_part;     
            prob = exp((prob - max(prob, [], 2) * ones(1, K)));

            sum_prob = prob * ones(K, 1);
            prob = prob ./ (sum_prob * ones(1, K));
                      
            alphavec = alpha + sum(prob, 1)';
            mvec = 0.5 * ((X .* (y * ones(1, n)))' * prob);
            for k=1:K
               cur_X = (sqrt(lambda_xi(:, k) .* prob(:, k)) * ones(1, n)) .* X;
               tilde_A{k} = A{k} + 2 * (cur_X' * cur_X);
               vvec(:, k) = (tilde_A{k} + 1e-10)\mvec(:, k);
            end

            L(iter, index) = count_mixture_evidence_vlb_L(X, y, A, alpha, alphavec, vvec, tilde_A, prob, xi);

            % M-step
            for k=1:K
               cur_matr = vvec(:, k) * vvec(:, k)' + inv(tilde_A{k} + 1e-10);
               for i=1:m
                  xi(i, k) = sqrt(X(i, :) * cur_matr * X(i, :)');
               end
               A{k} = diag(ones(n, 1) ./ (diag(cur_matr) + 1e-10)); 
            end
            to_stop = 1;
            for k=1:K
               if (sum(abs(diag(A{k}) - diag(A_old{k}))) >= a_eps * n)
                  to_stop = 0; 
                  break;
               end
               if (sum(abs(xi(:, k) - xi_old(:, k))) >= eps *m)
                  to_stop = 0;
                  break;
               end
            end
            if (to_stop == 1)
               break; 
            end
            A_old = A;
            xi_old = xi;
            %pi
        end
        L(iter + 1, index) = count_mixture_evidence_vlb_L(X, y, A, alpha, alphavec, vvec, tilde_A, prob, xi);
        A_best{index} = A;
        alphavec_best{index} = alphavec;
        vvec_best{index} = vvec;
        tilde_A_best{index} = tilde_A;
        prob_best{index} = prob;
        xi_best{index} = xi;
    end
    L_max = nanmax(L, [], 1);
    [~, idx_max] = max(L_max);
    A = A_best{idx_max};
    alphavec = alphavec_best{idx_max};
    vvec = vvec_best{idx_max};
    tilde_A = tilde_A_best{idx_max};
    prob = prob_best{idx_max};
    xi = xi_best{idx_max};
    
return