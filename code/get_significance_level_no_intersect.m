function t = get_significance_level_no_intersect(w1, hessian1, w2, hessian2)
    n = size(w1, 1);
    %size(w1), size(w2), size(hessian1), size(hessian2)
    s_score_neg_log = (w1 - w2)' * ((inv(hessian1 + 1e-10 * eye(n)) + inv(hessian2 + 1e-10 * eye(n)))\(w1 - w2));
    t = 1 - chi2cdf(s_score_neg_log, n);
%     if (t < 1e-3)
%        w1, w2 
%     end
return