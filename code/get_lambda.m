function lambda = get_lambda(xi)

    m = size(xi, 1);
    K = size(xi, 2);
    lambda = 0.5 * ones(m, K) ./ (xi + 1e-10) .* (ones(m, K) ./ (1 + exp(-xi))-0.5);

return