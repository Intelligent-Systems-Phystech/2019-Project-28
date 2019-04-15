function X_new = combine_copies_cliques(X, rho_0, fine_averaging, sigma_matrix)
    X_new = zscore(X);
    n = size(X, 2);
    m = size(X, 1);
    if (nargin < 4)
       sigma_matrix = X' * X; 
    end
    if (nargin < 3)
       fine_averaging = 0; 
    end
    corr_matrix = sigma_matrix ./ (ones(n, 1) * sqrt(diag(sigma_matrix))');
    corr_matrix = corr_matrix ./ (sqrt(diag(sigma_matrix)) * ones(1, n));
    
    corr_matrix = corr_matrix - eye(n);
    graph_matrix = (abs(corr_matrix) > rho_0);
    graph_matrix = graph_matrix - diag(diag(graph_matrix));
    if (max(max(graph_matrix)) == 0)
       return 
    end
    MC = maximalCliques(graph_matrix);
    num_vertices = sum(MC);
    clique_size = max(num_vertices);
    possible_cliques_idx = (num_vertices >= clique_size);
    possible_cliques = MC(:, possible_cliques_idx);
    
    index_best = 1;
    max_sum = -1;
    for i=1:size(possible_cliques, 2)
       cur_idx = (possible_cliques(:, i) == 1);
       %size(corr_matrix)
       cur_sum = sum(sum(abs(corr_matrix(cur_idx, cur_idx))));
       if (cur_sum > max_sum)
          max_sum = cur_sum;
          index_best = i;
       end
    end
    best_clique = possible_cliques(:, index_best);
    idx_best = 1:n;
    idx_best = idx_best(best_clique == 1);
    num_items = size(idx_best, 2);
    k = min(idx_best);
    mul = sign(corr_matrix(idx_best, k));
    mul(1) = 1;
    X_new(:, idx_best) = X_new(:, idx_best) .* (ones(m, 1) * mul');
    new_feat = X_new(:, idx_best) * ones(num_items, 1);
    new_feat = new_feat / std(new_feat);
    
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
    
    X_new(:, k) = new_feat;
    X_new(:, idx_best(2:size(idx_best, 2))) = [];
    
    X_new = combine_copies_cliques(X_new, rho_0);
    
return