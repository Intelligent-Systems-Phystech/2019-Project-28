% Illustration of asymptotic degeneracy of covariance matrix

m = [50, 100, 1000, 10000, 100000, 1000000];

w_1 = [1, 1]';
w_2 = [1, -1]';
X = randn(max(m), 2);
X = X * [1 0.7; 0 0.7];
y_1 = generate_single_logistic(X, w_1);
y_2 = generate_single_logistic(X, w_2);

for i=1:size(m, 2)
    m_cur = m(1, i);
    [A, ~] = maximize_evidence_single_logistic_laplace_non_diag(X(1:m_cur, :), y_1(1:m_cur));
    sigma = power(A(1, 1) * A(2, 2), 0.25);
    kappa = A(1, 2) / sigma^2;
    sigma_0 = sqrt(min(A(1, 1), A(2, 2)));
    m_cur, sigma_0, kappa
end
    

for i=1:size(m, 2)
    m_cur = m(1, i);
    [A, ~] = maximize_evidence_single_logistic_laplace_non_diag(X(1:m_cur, :), y_2(1:m_cur));
    sigma = power(A(1, 1) * A(2, 2), 0.25);
    kappa = A(1, 2) / sigma^2;
    sigma_0 = sqrt(min(A(1, 1), A(2, 2)));
    m_cur, sigma_0, kappa
end