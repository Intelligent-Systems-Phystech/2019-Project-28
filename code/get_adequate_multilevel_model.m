function [idx_new, mapping_new] = get_adequate_multilevel_model(X, y, A, idx, alpha, eps, lambda, max_iter, grad_eps, w0, mapping)
    if (nargin < 9)
       grad_eps = 1e-4; 
    end
    if (nargin < 8)
       max_iter = 20; 
    end
    if (nargin < 7)
       lambda = 1.; 
    end
    if (nargin < 6)
       eps = 1e-5; 
    end
    if (nargin < 5)
       alpha = 0.05; 
    end
    K = size(A, 1);
    if (nargin < 11)
       mapping = 1:K; 
    end
    if (nargin < 10)
       [w, hessian] = learn_multilevel_logistic(X, y, A, idx, eps, lambda, max_iter, grad_eps);
    else
       [w, hessian] = learn_multilevel_logistic(X, y, A, idx, eps, lambda, max_iter, grad_eps, w0);
    end
    t_matr = get_significance_matrix_no_intersect(w, hessian);
    
    t_matr = t_matr - eye(K);
    [~, index] = max(t_matr(:));
    [k, l] = ind2sub(size(t_matr), index);
    
    if (t_matr(k, l) < alpha)
       idx_new = idx;
       mapping_new = mapping;
       return 
    end
    
    k_copy = k;
    k = min(k, l);
    l = max(k_copy, l);
    
    A_new = A(1:(K-1), 1); % here we should in fact reoptimize A_{k}!!!
    
    idx_l = (idx == l);
    idx_last = (idx == K);
    
    mapping_new = mapping;
   
    for index=1:size(mapping, 2)
       if (mapping(1, index) == l)
           mapping_new(1, index) = k;
       end
    end
    
    idx_new = idx;
    idx_new(idx_l) = k;
    if (l ~= K)
        idx_new(idx_last) = l;
        A_new{l} = A{K};
        for index=1:size(mapping, 2)
           if (mapping(1, index) == K)
               mapping_new(1, index) = l;
           end
        end
    end 
    cur_idx_reopt = (idx_new == k);
    A_new{k} = maximize_evidence_single_logistic_laplace(X(cur_idx_reopt, :), y(cur_idx_reopt), A_new{k}, w{k}); % reoptimizing one model
    
    [idx_new, mapping_new] = get_adequate_multilevel_model(X, y, A_new, idx_new, alpha, eps, lambda, max_iter, grad_eps, w, mapping_new);
    %max(idx_new)
return