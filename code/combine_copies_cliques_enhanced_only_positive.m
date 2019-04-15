function [X_new, C_new] = combine_copies_cliques_enhanced_only_positive(X, rho_0, fine_averaging, corr_matrix_init, C, X_tr, sigma_matrix)   
    if (nargin < 6)
       X_tr = X; 
    end
    n = size(X_tr, 2);
    if (nargin < 5)
       C = eye(n); 
    end
    if (nargin < 4)
       corr_matrix_init = corr(X); 
    end
    if (nargin < 3)
       fine_averaging = 0; 
    end
    if (nargin < 7)
       sigma_matrix = X_tr' * X_tr; 
    end
    
    X_new = zscore(X_tr);
    C_new = C;
     
    corr_matrix = sigma_matrix ./ (ones(n, 1) * sqrt(diag(sigma_matrix))');
    corr_matrix = corr_matrix ./ (sqrt(diag(sigma_matrix)) * ones(1, n));
    
    corr_matrix = corr_matrix - eye(n);
    graph_matrix = ((corr_matrix) > rho_0);
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
       cur_sum = sum(sum((corr_matrix(cur_idx, cur_idx))));
       if (cur_sum > max_sum)
          max_sum = cur_sum;
          index_best = i;
       end
    end
    best_clique = possible_cliques(:, index_best);
    idx_best = 1:n;
    idx_best = idx_best(best_clique == 1);

    k = min(idx_best);
    [new_feat, C_common] = combine_correlated_features(X, C, idx_best, fine_averaging, corr_matrix_init);
    
    X_new(:, k) = new_feat;
    C_new(:, k) = C_common;
    X_new(:, idx_best(2:size(idx_best, 2))) = [];
    C_new(:, idx_best(2:size(idx_best, 2))) = [];
    
    [X_new, C_new] = combine_copies_cliques_enhanced(X, rho_0, fine_averaging, corr_matrix_init, C_new, X_new);
    
return