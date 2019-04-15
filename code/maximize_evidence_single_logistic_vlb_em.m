function A = maximize_evidence_single_logistic_vlb_em(X, y, A_0, eps, max_iter, gamma, a_eps)
    m = size(X, 1);
    n = size(X, 2);
    
    if (nargin < 7)
        a_eps = 1e-5;
    end
    if (nargin < 6)
       gamma = ones(m, 1); 
    end
    if (nargin < 5)
       max_iter = 20; 
    end
    if (nargin < 4)
       eps = 1e-5; 
    end   
    if (nargin < 3)
        A_0 = eye(n);
    end
    xi_0 = ones(m, 1);
    
    % Initialization
    xi = xi_0;
    A = A_0;
    xi_old = xi;
    A_old = A_0;
    
    for iter=1:max_iter
        % E-step
        v = 0.5 * X' * (gamma .* y);
        tilde_X = X .* (sqrt(gamma .* get_lambda(xi)) * ones(1, n));
        A_new = A + 2 * (tilde_X' * tilde_X);
        w_0 = A_new\v;
        % M-step
        conv_matr = (w_0 * w_0' + inv(A_new));
        for i=1:m
           xi(i) = sqrt(X(i, :) * conv_matr * X(i, :)');  % think if possible to make efficiently in matrices
        end
        A = diag(ones(n, 1) ./ diag(conv_matr)); % solution in diagonal matrices
        if (and(sum(abs(diag(A) - diag(A_old))) < a_eps * n, sum(abs(xi - xi_old)) < eps * m))
            break
        end
        A_old = A;
        xi_old = xi;
    end

return