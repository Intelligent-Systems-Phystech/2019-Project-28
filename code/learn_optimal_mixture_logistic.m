function [w, pi, hessian, A, L] = learn_optimal_mixture_logistic(X, y, K, alpha, eps, a_eps, em_max_iter, pi_eps, num_start, lambda, max_iter, grad_eps, pi_0, A_0, w_0) %, w_evol, pi_evol, L_evol, L_evol1]
    slight_opt = 0;
    if (nargin < 12)
       grad_eps = 1e-4; 
    end
    if (nargin < 11)
       max_iter = 10; 
    end
    if (nargin < 10)
       lambda = 1.; 
    end
    if (nargin < 9)
       num_start = 10; 
    end
    if (nargin <8)
        pi_eps = 1e-10;
    end
    if (nargin < 7)
       em_max_iter = 10; 
    end
    if (nargin < 6)
       a_eps = 1e-5; 
    end
    if (nargin < 5)
       eps = 1e-5; 
    end
    if (nargin >= 15)
       num_start = 1;
       slight_opt = 1;
    end
    
    m = size(X, 1);
    n = size(X, 2);
    
    L = nan * ones(em_max_iter, num_start);
    pi_best = cell(num_start, 1);
    w_best = cell(num_start, 1);
    hessian_best = cell(num_start, 1);
    A_best = cell(num_start, 1);
%     w_evol = zeros(n, K, em_max_iter);
%     pi_evol = zeros(n, em_max_iter);
%     L_evol = zeros(max_iter, em_max_iter);
%     L_evol1 = zeros(max_iter, em_max_iter);
      
    for index=1:num_start
        %disp "START", index
        % Initialization
        if (slight_opt == 0)
            if (or(rand(1) > 0.5, index <= 2))
                pi = rand(K, 1);
            else
                pi = gamrnd(0.1 * ones(K, 1), ones(K, 1)); % making slightly sparse dirichlet sampling
            end
            pi = pi / sum(pi);
            w = randn(n, K);
            A = cell(K, 1);
            for k=1:K
               A{k} = diag(10 * abs(randn(n, 1)));
            end
        else
           pi = pi_0;
           w = w_0;
           A = A_0;
        end  
        %pi = [0.99, 0.01]';
        pi_old = pi;
        
        hessian = cell(K, 1);
        is_ok = ones(K, 1);
        
        A_old = A;
        w_old = w;
        
        %L_cur_old = nan * ones(max_iter + 1, K);
        
        for i=1:em_max_iter
            %i
            % E-step
            sigma = ones(m, K) ./ (1 + exp(-(X * w) .* (y * ones(1, K))));
            %size(sigma), size(pi), m
            prob = sigma .* (ones(m, 1) * pi');
            sum_prob = prob * ones(K, 1);
            prob = prob ./ (sum_prob * ones(1, K));

            % M-step
            L_cur = nan * ones(max_iter, K);

            for k=1:K
               if (is_ok(k))
                   [A{k}, w(:, k), hessian{k}, L_cur(:, k)] = maximize_evidence_single_logistic_laplace(X, y, A{k}, w(:, k), eps, lambda, max_iter, grad_eps, prob(:, k), a_eps);
%                    A{k} = maximize_evidence_single_logistic_vlb_em(X, y, A{k}, eps, max_iter, prob(:, k), a_eps);
%                    [w(:, k), hessian{k}] = learn_single_logistic(X, y, A{k}, eps, lambda, max_iter, grad_eps, prob(:, k), w(:, k));
%                    L_cur(:, k) = count_evidence_single_logistic(A{k}, w(:, k), hessian{k}, X, y, prob(:, k)) * ones(max_iter, 1);
                     % It was the same as the one with Laplace. Looks like due to the change in gamma should not necessarily be increasing (looks strange).
               else
                   L_cur(:, k) = zeros(max_iter, 1); %count_evidence_single_logistic(A{k}, w(:, k), hessian{k}, X, y, zeros(m, 1)) * ones(max_iter, 1);
               end
            end
%             L_evol1(:, i) = L_cur(:, 1);
%             L_evol(:, i) = L_cur(:, 2);
            
%             if (nanmin(nanmin(L_cur(2:max_iter, :) - L_cur(1:(max_iter - 1), :))) < -0.02)
%                 pi, L_cur, is_ok, sum(prob)
%             end
            pi = alpha - 1 + sum(prob, 1)';
            is_ok(pi <= 0) = 0;
            pi = max(pi, 0);
            pi = pi / sum(pi);
            %sum(isnan(nanmax(L_cur, [], 1)))
            L_pi = gammaln(K * alpha) - K * gammaln(alpha) + sum(gammaln(alpha + sum(prob, 1)')) - gammaln(K * alpha + m);
            L(i, index) = L_pi + nansum(nanmax(L_cur, [], 1)) - sum(sum(prob .* log(prob + 1e-10)));
            
%             w_evol(:, :, i) = w;
%             pi_evol(:, i) = pi;
            
            to_stop = 1;
            if (sum(abs(pi - pi_old)) >= pi_eps)
               to_stop = 0;
            end
            for k=1:K
               if (sum(abs(diag(A{k}) - diag(A_old{k}))) >= a_eps * n)
                  to_stop = 0; 
                  break;
               end
               if (sum(abs(w(:, k) - w_old(:, k))) >= eps)
                  to_stop = 0;
                  break;
               end
            end
            if (to_stop == 1)
               break; 
            end
            w_old = w;
            A_old = A;
            %pi
        end
        pi_best{index} = pi;
        w_best{index} = w;
        hessian_best{index} = hessian;
        A_best{index} = A;
    end
    L_max = nanmax(L, [], 1);
    [~, idx_max] = max(L_max);
    w = w_best{idx_max};
    pi = pi_best{idx_max};
    hessian = hessian_best{idx_max};
    A = A_best{idx_max};
    cell2mat(pi_best)
    
return