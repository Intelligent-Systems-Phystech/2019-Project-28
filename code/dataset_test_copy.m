X = dlmread('white_whine_data.txt');
y = X(:, size(X, 2));
y(y <= 5) = -1;
y(y >= 6) = 1;
X(:, size(X, 2)) = [];
X = zscore(X);
X = [ones(size(X, 1), 1) X];

m = size(X, 1);
n = size(X, 2);

learn_fraction = 0.7;
init_max_iter = 5000;

[idx_learn_all, idx_test_all] = make_split(m, learn_fraction, init_max_iter);
dlmwrite('wine_idx_learn.csv', idx_learn_all);
dlmwrite('wine_idx_test.csv', idx_test_all);

alpha_init = 0.001;
K_all = [2, 3, 5, 10, 20, 50];
max_iter = 200;
AUC_joint = zeros(max_iter, size(K_all, 2));
AUC_adequate_joint = zeros(max_iter, size(K_all, 2));
AUC_common_joint = zeros(max_iter, size(K_all, 2));
mappings_all = cell(size(K_all, 2), 1);

tic;
for index=1:size(K_all, 2)
    K = K_all(1, index);
    alpha = alpha_init / K * 2;
    idx = kmeans(X, K);

    m_all = zeros(K, 1);
    idx_all = cell(K, 1);
    for k=1:K
       cur_idx = 1:m;
       cur_idx = cur_idx(idx == k);
       idx_all{k} = cur_idx;
       m_all(k) = size(cur_idx, 2);
    end

    %X(:, 1) = [];   
    mappings = zeros(max_iter, K);

    for iter=1:max_iter
       iter
       idx_learn = idx_learn_all(:, iter);
       idx_test = idx_test_all(:, iter);
       [A, w, hessian] = maximize_evidence_multilevel_logistic_laplace(X(idx_learn, :), y(idx_learn), idx(idx_learn));
       [~, AUC_joint(iter, index)] = test_multilevel_logistic(X(idx_test, :), y(idx_test), w, idx(idx_test));
       [idx_new, mapping_new] = get_adequate_multilevel_model(X(idx_learn, :), y(idx_learn), A, idx(idx_learn), alpha);
       mappings(iter, :) = mapping_new;
       [~, w_adequate, ~] = maximize_evidence_multilevel_logistic_laplace(X(idx_learn, :), y(idx_learn), idx_new);
       [~, w_common, ~, ~] = maximize_evidence_single_logistic_laplace(X(idx_learn, :), y(idx_learn));
       AUC_common_joint(iter, index) = test_single_logistic(X(idx_test, :), y(idx_test), w_common);
       [~, AUC_adequate_joint(iter, index)] = test_multilevel_logistic(X(idx_test, :), y(idx_test), w_adequate, mapping_new(idx(idx_test))');
    end
    mappings_all{index} = mappings;
end

AUC_diff = AUC_adequate_joint - AUC_joint;
mean_diff = nanmean(AUC_diff);
std_diff = nanstd(AUC_diff);

t_stat = mean_diff ./ (std_diff + 1e-10) * sqrt(max_iter);

AUC_diff_common = AUC_adequate_joint - AUC_common_joint;
mean_diff_common = nanmean(AUC_diff_common);
std_diff_common = nanstd(AUC_diff_common);

t_stat_common = mean_diff_common ./ (std_diff_common + 1e-10) * sqrt(max_iter);

mean_AUC_adequate_joint = nanmean(AUC_adequate_joint);
mean_AUC_common_joint = nanmean(AUC_common_joint);
mean_AUC_joint = nanmean(AUC_joint);
toc;
