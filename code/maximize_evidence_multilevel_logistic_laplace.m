function [A, w, hessian] = maximize_evidence_multilevel_logistic_laplace(X, y, idx, A_0, w_0, eps, lambda, max_iter, grad_eps, a_eps) 
% A is a cell array of size K
% idx contains a split of objects among the models
n = size(X, 2);
K = max(idx);

if (nargin < 10)
    a_eps = 1e-4;
end
if (nargin < 9)
   grad_eps = 1e-4; 
end
if (nargin < 8)
   max_iter = 100; 
end
if (nargin < 7)
   lambda = 1.; 
end
if (nargin < 6)
   eps = 1e-5; 
end   
if (nargin < 5)
    w_0 = cell(K, 1);
    for k=1:K
        w_0{k} = zeros(n, 1);
    end
end
if (nargin < 4)
    A_0 = cell(K, 1);
    for k=1:K
        A_0{k} = eye(n);
    end
end

w = cell(K, 1);
hessian = cell(K, 1);
A = cell(K, 1);

for k=1:K
    cur_idx = (idx == k);
    cur_m = sum(cur_idx);
    
    if (cur_m >= 10)
        [A{k}, w{k}, hessian{k}] = maximize_evidence_single_logistic_laplace(X(cur_idx, :), y(cur_idx), A_0{k}, w_0{k}, eps, lambda, max_iter, grad_eps, ones(cur_m, 1), a_eps);
    else
       A{k} = A_0{k};
       w{k} = w_0{k};
       hessian{k} = A_0{k} + eye(n);
    end
end


return