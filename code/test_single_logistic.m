function AUC = test_single_logistic(X, y, w)
    m = size(X, 1);
    prob = ones(m, 1) ./ (1 + exp(-X * w));
    try
       [~, ~, ~, AUC] = perfcurve(y, prob, 1);
    catch
       AUC = nan; 
    end
return