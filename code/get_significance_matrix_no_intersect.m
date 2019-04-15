function t_matr = get_significance_matrix_no_intersect(w, hessian, k_recalc)
    if (nargin < 3)
       k_recalc = -1; 
    end
    K = size(w, 1);
    t_matr = ones(K, K);
    for k=1:(K-1)
        for l=(k + 1):K
            if (or(or(k == k_recalc, l == k_recalc), k_recalc <= 0))
                t_matr(k, l) = get_significance_level_no_intersect(w{k}, hessian{k}, w{l}, hessian{l});
                t_matr(l, k) = t_matr(k, l);
            end
        end
    end
return