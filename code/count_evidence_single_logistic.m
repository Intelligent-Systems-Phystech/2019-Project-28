function L = count_evidence_single_logistic(A, w, hessian, X, y, gamma)
    m = size(X, 1);
    L_w = sum(gamma .* log(ones(m, 1) ./ (1 + exp(-y.* (X * w)))));
    L_A = 0.5 * sum(log(diag(A) + 1e-10));
    L_H = 0.5 * sum(log(eig(hessian) + 1e-10));
    L = L_A - 0.5 * w' * A * w + L_w - L_H; %0.5 * log(det(hessian) + 1e-10);

return