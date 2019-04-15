X = dlmread('housing.csv');
y = 2 * X(:, size(X, 2)) - 1;

X(:, size(X, 2)) = [];
X = zscore(X);
X = [ones(size(X, 1), 1) X];

m = size(X, 1);
n = size(X, 2);

learn_fraction = 0.7;
init_max_iter = 5000;
 
% [idx_learn_all, idx_test_all] = make_split(m, learn_fraction, init_max_iter);
% dlmwrite('housing_idx_learn.csv', idx_learn_all);
% dlmwrite('housing_idx_test.csv', idx_test_all);
idx_learn_all = dlmread('housing_idx_learn.csv');
idx_test_all = dlmread('housing_idx_test.csv');

% K_init_all = [2, 3, 5, 10, 20, 50];
% indices = zeros(m, size(K_init_all, 2));
% for index=1:size(K_init_all, 2)
%    indices(:, index) = kmeans(X, K_init_all(index)); 
% end
% dlmwrite('housing_indices_kmeans.csv', indices);
indices = dlmread('housing_indices_kmeans.csv');

max_iter = 50;
alpha_init = 0.01;
K_all = [2, 3, 5, 10, 20];
AUC_joint = zeros(max_iter, size(K_all, 2));
AUC_adequate_joint = zeros(max_iter, size(K_all, 2));

AUC_mix_joint = zeros(max_iter, size(K_all, 2));
AUC_mix_adequate_joint = zeros(max_iter, size(K_all, 2));
mix_K = zeros(max_iter, size(K_all, 2));
mix_K_init = zeros(max_iter, size(K_all, 2));

AUC_mix_sparse_joint = zeros(max_iter, 1);
AUC_mix_adequate_sparse_joint = zeros(max_iter, 1);
mix_K_sparse = zeros(max_iter, 1);
mix_K_init_sparse = zeros(max_iter, 1);

AUC_common_joint = zeros(max_iter, size(K_all, 2));
mappings_all = cell(size(K_all, 2), 1);

tic;
for index=1:size(K_all, 2)
    disp "INDEX", index
    K = K_all(1, index);
    alpha = alpha_init / K * 2;
    idx = indices(:, index);%kmeans(X, K);

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
       % Multilevel
       [A, w, ~] = maximize_evidence_multilevel_logistic_laplace(X(idx_learn, :), y(idx_learn), idx(idx_learn));
       [~, AUC_joint(iter, index)] = test_multilevel_logistic(X(idx_test, :), y(idx_test), w, idx(idx_test));
       [idx_new, mapping_new] = get_adequate_multilevel_model(X(idx_learn, :), y(idx_learn), A, idx(idx_learn), alpha);
       mappings(iter, :) = mapping_new;
       [~, w_adequate, ~] = maximize_evidence_multilevel_logistic_laplace(X(idx_learn, :), y(idx_learn), idx_new);
       [~, AUC_adequate_joint(iter, index)] = test_multilevel_logistic(X(idx_test, :), y(idx_test), w_adequate, mapping_new(idx(idx_test))');
       
       %Mixture
       [w_mix, pi, hessian_mix, A_mix, ~] = learn_optimal_mixture_logistic(X(idx_learn, :), y(idx_learn), K, 1.);
       [w_mix, pi, hessian_mix, A_mix] = filter_mixture(w_mix, pi, hessian_mix, A_mix);
       AUC_mix_joint(iter, index) = test_mixture_logistic(X(idx_test, :), y(idx_test), pi, w_mix);
       [w_mix_adeq, pi_adeq, hessian_mix_adeq, A_mix_adeq] = get_adequate_mixture_model(X(idx_learn, :), y(idx_learn), size(pi, 1), 1., alpha, pi, w_mix, A_mix, hessian_mix);
       AUC_mix_adequate_joint(iter, index) = test_mixture_logistic(X(idx_test, :), y(idx_test), pi_adeq, w_mix_adeq);
       
       mix_K(iter, index) = sum(pi_adeq > 1e-3);
       mix_K_init(iter, index) = sum(pi > 1e-3);
       
       % Sparse mixture
       if (index == 1)
           [w_mix, pi, hessian_mix, A_mix, ~] = learn_optimal_mixture_logistic(X(idx_learn, :), y(idx_learn), 200, 1e-6);
           [w_mix, pi, hessian_mix, A_mix] = filter_mixture(w_mix, pi, hessian_mix, A_mix);
           AUC_mix_sparse_joint(iter, 1) = test_mixture_logistic(X(idx_test, :), y(idx_test), pi, w_mix);
           [w_mix_adeq, pi_adeq, hessian_mix_adeq, A_mix_adeq] = get_adequate_mixture_model(X(idx_learn, :), y(idx_learn), size(pi, 1), 1., alpha, pi, w_mix, A_mix, hessian_mix);
           AUC_mix_adequate_sparse_joint(iter, 1) = test_mixture_logistic(X(idx_test, :), y(idx_test), pi_adeq, w_mix_adeq);
           
           mix_K_sparse(iter, 1) = sum(pi_adeq > 1e-3);
           mix_K_init_sparse(iter, 1) = sum(pi > 1e-3);
       end
       
       % Single model
       [~, w_common, ~, ~] = maximize_evidence_single_logistic_laplace(X(idx_learn, :), y(idx_learn));
       AUC_common_joint(iter, index) = test_single_logistic(X(idx_test, :), y(idx_test), w_common);
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

% Multilevel number of groups

multilevel_K = zeros(max_iter, size(K_all, 2));
for index=1:size(K_all, 2)
    cur_mapping = mappings_all{index};
    multilevel_K(:, index) = max(cur_mapping, [], 2);
end
mean_multilevel_K = mean(multilevel_K);

% Mixture comparison

AUC_mix_diff = AUC_mix_adequate_joint - AUC_mix_joint;
mean_diff = nanmean(AUC_mix_diff);
std_diff = nanstd(AUC_mix_diff);

t_stat_mix = mean_diff ./ (std_diff + 1e-10) * sqrt(max_iter);

AUC_mix_diff_common = AUC_mix_adequate_joint - AUC_common_joint;
mean_diff_common = nanmean(AUC_mix_diff_common);
std_diff_common = nanstd(AUC_mix_diff_common);

t_stat_mix_common = mean_diff_common ./ (std_diff_common + 1e-10) * sqrt(max_iter);

mean_AUC_adequate_joint = nanmean(AUC_adequate_joint);
mean_AUC_common_joint = nanmean(AUC_common_joint);
mean_AUC_joint = nanmean(AUC_joint);
mean_AUC_mix_adequate_joint = nanmean(AUC_mix_adequate_joint);
mean_AUC_mix_joint = nanmean(AUC_mix_joint);

mean_K_mix = mean(mix_K);
mean_K_mix_init = mean(mix_K_init);

% Sparse comparison

AUC_mix_sparse_diff = AUC_mix_adequate_sparse_joint - AUC_mix_sparse_joint;
mean_diff = nanmean(AUC_mix_sparse_diff);
std_diff = nanstd(AUC_mix_sparse_diff);

t_stat_mix_sparse = mean_diff ./ (std_diff + 1e-10) * sqrt(max_iter);

mean_AUC_mix_adequate_sparse_joint = nanmean(AUC_mix_adequate_sparse_joint);
mean_AUC_mix_sparse_joint = nanmean(AUC_mix_sparse_joint);
mean_K_mix_sparse = mean(mix_K_sparse);
mean_K_mix_init_sparse = mean(mix_K_init_sparse);
toc;
