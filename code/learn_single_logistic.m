function [w, hessian, grad] = learn_single_logistic(X, y, A, eps, lambda, max_iter, grad_eps, gamma, w_0)
    if (nargin < 7)
       grad_eps = 1e-10; 
    end
    if (nargin < 6)
       max_iter = 20; 
    end
    if (nargin < 5)
       lambda = 1.; 
    end
    if (nargin < 4)
       eps = 1e-5; 
    end
    m = size(X, 1);
    if (nargin < 8)
       gamma = ones(m, 1); 
    end
    n = size(X, 2);

    if (nargin < 9)
        w_0 = zeros(n, 1); % randn(n, 1);
    end
    w_old = w_0;
    w = w_0;
    %lambda_cur = lambda;
    for iter=1:max_iter
       %iter
       sigma = ones(m, 1) ./ (1 + exp((X * w) .* y));
       grad = A * w - X' * (sigma .* y .* gamma);
       %hessian_1 = A + X' * diag(sigma .* (1 - sigma) .* gamma) * X;
       tilde_X = X .* (sqrt(sigma .* (1 - sigma) .* gamma) * ones(1, n));
       hessian = A + tilde_X' * tilde_X;
       %max(abs(hessian - hessian_1))
       
       step = -(hessian + 1e-10 * eye(n)) \ grad;
       step = step / max(1, sum(abs(step)));
       
       if and(sqrt((w - w_old)' * (w - w_old)) < eps, sqrt(grad' * grad) < grad_eps)
          break; 
       end
       w = w + lambda * step;
       w_old = w;
    end
return