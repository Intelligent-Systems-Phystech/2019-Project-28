% Showing possible inadequacy of a mixture
% (it is always possible to build adequate mixture (just use 1 model), but
% the quality might suffer from this

m = 5000;
m_test = 20000;
n = 2;
K = 10;
X = randn(m, n);
X_test = randn(m_test, n);

w_0 = randn(n, 1);
rho = 0.;

% w_init = rho * w_0 * ones(1, K) + sqrt(1 - rho^2) * randn(n, K);
w_init = (2 * 3.14159 / K) * linspace(0, K - 1, K);
w_init = [cos(w_init); sin(w_init)];
%weights = 5 * abs(randn(1, K));
weights = 5 * sqrt(rand(1, K));
w_init = w_init .* [weights; weights];
pi_init = 1 / K * ones(K, 1);

y_test = generate_mixture_logistic(X_test, w_init, pi_init);
AUC_best = test_mixture_logistic(X_test, y_test, pi_init, w_init);

min_dist = 1e10;
for i=1:(K-1)
    for j=(i+1):K
        min_dist = min(min_dist, norm(w_init(:, i) - w_init(:, j)));
        %min_dist = min(min_dist, norm(w_init(:, i) + w_init(:, j)));
    end
end


num_start = 20; 

pi_eps = 1e-5;
em_max_iter = 100; 
grad_eps = 1e-4; 
max_iter = 20; 
lambda = 1.; 
eps = 1e-5; 
signif_alpha = 0.05; 


num_iter = 100;
min_dist_w = zeros(num_iter, 1);
t_matr_evol = cell(num_iter, 1);
pi_evol = zeros(K, num_iter);
w_evol = cell(num_iter, 1);
AUC_init = zeros(num_iter, 1);
AUC_adequate = zeros(num_iter, 1);

for iter=1:num_iter

    y = generate_mixture_logistic(X, w_init, 1 / K * ones(K, 1));

    A = cell(K, 1);
    for k=1:K
       A{k} = zeros(n, n); 
    end
    [w, pi, hessian] = learn_mixture_logistic(X, y, A, 1.);
    pi_evol(:, iter) = pi;
    w_evol{iter} = w;
    
    AUC_init(iter, 1) = test_mixture_logistic(X_test, y_test, pi, w);

    min_dist_w(iter, 1) = 1e10;
    for i=1:(K-1)
        for j=(i+1):K
            min_dist_w(iter, 1) = min(min_dist_w(iter, 1), norm(w(:, i) - w(:, j)));
        end
    end

    t_matr = ones(K, K);
    for i=1:(K-1)
        for j=(i+1):K
            t_matr(i, j) = get_significance_level_no_intersect(w(:, i), hessian{i}, w(:, j), hessian{j});
            t_matr(j, i) = t_matr(i, j);
        end
    end
    
    t_matr_evol{iter} = t_matr;

    [w, pi, hessian] = get_adequate_mixture_model_no_A(X, y, K, signif_alpha, eps, lambda, max_iter, grad_eps, em_max_iter, pi_eps, num_start, pi, w);
    AUC_adequate(iter, 1) = test_mixture_logistic(X_test, y_test, pi, w);

end

