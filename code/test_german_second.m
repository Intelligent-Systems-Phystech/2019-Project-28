X = dlmread('german_data.csv'); % No heterogeneity be term
y = X(:, size(X, 2));
y = 2 * y - 1;
X(:, size(X, 2)) = [];

term = X(:, 3);
m = size(X, 1);
n = size(X, 2);
idx = 2 * ones(m, 1);
idx(term >= 24) = 3;
idx(term <= 12) = 1;
K = 3;

m_all = zeros(K, 1);
idx_all = cell(K, 1);
for k=1:K
   cur_idx = 1:m;
   cur_idx = cur_idx(idx == k);
   idx_all{k} = cur_idx;
   m_all(k) = size(cur_idx, 2);
end

X(:, 2:n) = zscore(X(:, 2:n));
%X(:, 1) = [];

tic;
learn_fraction = 0.7;
max_iter = 50;
AUC = zeros(max_iter, K);
AUC_joint = zeros(max_iter, 1);
AUC_adequate = zeros(max_iter, K);
AUC_adequate_joint = zeros(max_iter, 1);
AUC_common = zeros(max_iter, K);
AUC_common_joint = zeros(max_iter, 1);

mappings = zeros(max_iter, K);
alpha = 0.001;

for iter=1:max_iter
   iter
   idx_learn = [];
   idx_test = [];
   for k=1:K
        cur_perm = randperm(m_all(k));
        cur_end_learn = round(learn_fraction * m_all(k));
        idx_learn = [idx_learn, idx_all{k}(cur_perm(1:cur_end_learn))];
        idx_test = [idx_test, idx_all{k}(cur_perm((cur_end_learn + 1):m_all(k)))];
   end
   [A, w, hessian] = maximize_evidence_multilevel_logistic_laplace(X(idx_learn, :), y(idx_learn), idx(idx_learn));
   [AUC(iter, :), AUC_joint(iter, 1)] = test_multilevel_logistic(X(idx_test, :), y(idx_test), w, idx(idx_test));
   [idx_new, mapping_new] = get_adequate_multilevel_model(X(idx_learn, :), y(idx_learn), A, idx(idx_learn), alpha);
   mappings(iter, :) = mapping_new;
   [~, w_adequate, ~] = maximize_evidence_multilevel_logistic_laplace(X(idx_learn, :), y(idx_learn), idx_new);
   [~, w_common, ~, ~] = maximize_evidence_single_logistic_laplace(X(idx_learn, :), y(idx_learn));
   for k=1:K
      cur_idx = intersect(idx_test, idx_all{k});
      AUC_adequate(iter, k) = test_single_logistic(X(cur_idx, :), y(cur_idx), w_adequate{mapping_new(k)});
      AUC_common(iter, k) = test_single_logistic(X(cur_idx, :), y(cur_idx), w_common);
   end
   AUC_common_joint(iter, 1) = test_single_logistic(X(idx_test, :), y(idx_test), w_common);
   [~, AUC_adequate_joint(iter, 1)] = test_multilevel_logistic(X(idx_test, :), y(idx_test), w_adequate, mapping_new(idx(idx_test))');
end

AUC_diff = AUC_adequate - AUC;
mean_diff = mean(AUC_diff);
std_diff = std(AUC_diff);

t_stat = mean_diff ./ std_diff * sqrt(max_iter);

AUC_diff_common = AUC_adequate - AUC_common;
mean_diff_common = mean(AUC_diff_common);
std_diff_common = std(AUC_diff_common);

t_stat_common = mean_diff_common ./ std_diff_common * sqrt(max_iter);
toc;
