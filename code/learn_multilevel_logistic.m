function [w, hessian] = learn_multilevel_logistic(X, y, A, idx, eps, lambda, max_iter, grad_eps, w0) 
% A is a cell array of size K
% idx contains a split of objects among the models
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
K = size(A, 1);
w = cell(K, 1);
hessian = cell(K, 1);
m = size(X, 1);

for k=1:K
    cur_idx = (idx == k);
    cur_m = sum(cur_idx);
    if (nargin < 9)
        [w{k}, hessian{k}] = learn_single_logistic(X(cur_idx, :), y(cur_idx), A{k}, eps, lambda, max_iter, grad_eps, ones(cur_m, 1));
    else
        [w{k}, hessian{k}] = learn_single_logistic(X(cur_idx, :), y(cur_idx), A{k}, eps, lambda, max_iter, grad_eps, ones(cur_m, 1), w0{k});
    end
end


return