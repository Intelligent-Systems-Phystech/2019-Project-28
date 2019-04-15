function [new_feat, C_common] = combine_correlated_features(X, C, idx_best_init, fine_averaging, corr_matrix)  
    m = size(X, 1);
    n = size(X, 2);
    C_common = sum(C(:, idx_best_init), 2);
    idx_best = 1:n;
    idx_best = idx_best(C_common == 1);
    num_items = size(idx_best, 2);
    k = min(idx_best);
    mul = sign(corr_matrix(idx_best, k));
    mul(1) = 1;
    X_new = X;
    X_new(:, idx_best) = X_new(:, idx_best) .* (ones(m, 1) * mul');
    new_feat = X_new(:, idx_best) * ones(num_items, 1);
    new_feat = zscore(new_feat);
    
    if (fine_averaging)
       delta = 1e-5;
       max_iter = 20;
       new_feat_old = new_feat;
       for i=1:max_iter
           diff = X_new(:, idx_best) - new_feat * ones(1, num_items);
           d_sq = sum(diff .* diff) / m;
           sigma_est_sq = max(0.01, 0.5 * d_sq .* (2 - 0.5 * d_sq));
           v = ones(1, num_items) ./ sigma_est_sq;
           new_feat = X_new(:, idx_best) * v';
           new_feat = new_feat / std(new_feat);
           if (sum(abs(new_feat - new_feat_old)) < n * delta)
               break;
           end
           new_feat_old = new_feat;
       end
    end
    
return