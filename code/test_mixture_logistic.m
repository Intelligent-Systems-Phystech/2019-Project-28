function AUC = test_mixture_logistic(X, y, pi, w)
    m = size(X, 1);
    K = size(pi, 1);

    sigma = ones(m, K) ./ (1 + exp(-X * w));
    prob = sigma .* (ones(m, 1) * pi');
    prob = sum(prob, 2);
    try
        [~, ~, ~, AUC] = perfcurve(y, prob, 1);
    catch
       AUC = nan; 
    end

return