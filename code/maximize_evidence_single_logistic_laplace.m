function [A, w, hessian, L] = maximize_evidence_single_logistic_laplace(X, y, A_0, w_0, eps, lambda, max_iter, grad_eps, gamma, a_eps)
    m = size(X, 1);
    n = size(X, 2);
    
    if (nargin < 10)
        a_eps = 1e-4;
    end
    if (nargin < 9)
       gamma = ones(m, 1); 
    end
    if (nargin < 8)
       grad_eps = 1e-4; 
    end
    if (nargin < 7)
       max_iter = 50; 
    end
    if (nargin < 6)
       lambda = 1.; 
    end
    if (nargin < 5)
       eps = 1e-5; 
    end   
    if (nargin < 4)
        w_0 = zeros(n, 1); % randn(n, 1);
    end
    if (nargin < 3)
        A_0 = eye(n);
    end
    
    L = nan * ones(max_iter, 1);
    
    % Initialization
    w = w_0;
    A = A_0;
    w_old = w;
    A_old = A_0;
       
    for i=1:max_iter
        [w, hessian, grad] = learn_single_logistic(X, y, A, eps, lambda, 3, grad_eps, gamma, w); % not full optimization but several steps
        L(i, 1) = count_evidence_single_logistic(A, w, hessian, X, y, gamma);
        A = (ones(n, 1) - diag(A) .* diag(inv(hessian + 1e-10 * eye(n)))) ./ (w .* w + 1e-10); 
        %A(A > 1e3) = 1e10;
        A = diag(A);
        A = min(A, 1e5);
        if and(and(sqrt((w - w_old)' * (w - w_old)) < eps, sqrt(grad' * grad) < grad_eps), sum(abs(diag(A) - diag(A_old))) < a_eps * n)
          break; 
        end
        w_old = w;
        A_old = A;
    end
    if (i ~= max_iter)
        L(i + 1, 1) = count_evidence_single_logistic(A, w, hessian, X, y, gamma);
        if (L(i + 1, 1) < L(i, 1) - 1e-3)
           w, A, L(i + 1, 1), L(i, 1), hessian
        end
    end
    w(diag(A) >= 1e5) = 0;


return