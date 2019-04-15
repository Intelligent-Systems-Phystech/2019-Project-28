function [idx_learn_all, idx_test_all] = make_split(m, learn_fraction, max_iter)
    learn_size = round(learn_fraction * m);
    test_size = m - learn_size;
    idx_learn_all = zeros(learn_size, max_iter);
    idx_test_all = zeros(test_size, max_iter);
    for iter=1:max_iter
            cur_perm = randperm(m);
            idx_learn_all(:, iter) = cur_perm(1:learn_size)';
            idx_test_all(:, iter) = cur_perm((learn_size + 1):m)';
    end
return