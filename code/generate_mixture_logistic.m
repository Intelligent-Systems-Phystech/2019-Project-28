function y = generate_mixture_logistic(X, w, pi)
% w is a matrix of size n * K
% pi contains model weights   
    m = size(X, 1);
    K = size(pi, 1);

    sigma = ones(m, K) ./ (1 + exp(-X * w));
    prob = sigma .* (ones(m, 1) * pi');
    prob = sum(prob, 2);
    y = 2 * (rand(m, 1) < prob) - 1;
return