function X_new = combine_copies(X, rho_0, sigma_matrix)
    X_new = X;
    n = size(X, 2);
    if (nargin < 3)
       sigma_matrix = X' * X; 
    end
    corr_matrix = sigma_matrix ./ (ones(n, 1) * sqrt(diag(sigma_matrix))');
    corr_matrix = corr_matrix ./ (sqrt(diag(sigma_matrix)) * ones(1, n));
    
    corr_matrix = corr_matrix - eye(n);
    [~, index] = max(abs(corr_matrix(:)));
    [k, l] = ind2sub(size(corr_matrix), index);
    if (abs(corr_matrix(k, l)) < rho_0)
       return 
    end
    k_copy = k;
    k = min(k, l);
    l = max(k_copy, l);
    
    X_new(:, l) = X_new(:, l) * sign(corr_matrix(k, l));
    
    X_new(:, k) = X_new(:, k) + X_new(:, l);
    X_new(:, l) = [];
    X_new(:, k) = X_new(:, k) / std(X_new(:, k));
    X_new = combine_copies(X_new, rho_0);
return