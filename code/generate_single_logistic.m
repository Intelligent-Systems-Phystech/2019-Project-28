function y = generate_single_logistic(X, w)
    m = size(X, 1);
    p = ones(m, 1) ./ (1 + exp(-X * w));
    y = 2 * (rand(m, 1) < p) - 1;
return