m = 5000;
n = 2;
X = randn(m, n);
w = [1, 1]';

y = generate_single_logistic(X, w);


K = 1000;
sample_weights = zeros(m, K); % objects' weights
for k=1:K
   cur_sample_idx = randsample(m, m, true);
   occurences = hist(cur_sample_idx, 1:m);
   sample_weights(:, k) = occurences';
end

w = zeros(n, K);
hessian = cell(K, 1);

for k=1:K
    k
    [A, cur_w, cur_hessian, L] = maximize_evidence_single_logistic_laplace(X, y, eye(n), zeros(n, 1), 1e-5, 1, 20, 1e-10, sample_weights(:, k));
    cur_w(diag(A) >= 1e5) = 0;
    w(:, k) = cur_w;
    hessian{k} = cur_hessian;
end


t_matr = ones(K, K);

for k=1:K
    k
    for l=k:K
        t_matr(k, l) = get_significance_level_no_intersect(w(:, k), hessian{k}, w(:, l), hessian{l});
        t_matr(l, k) = t_matr(k, l);
    end
end

h=figure;
hold('on');
imagesc(t_matr);
set(gca, 'FontSize', 24, 'FontName', 'Times');

xlabel('$k$','FontSize', 24, 'Interpreter', 'latex');
ylabel('$l$','FontSize',24, 'Interpreter', 'latex');


axis 'tight' 'square'
bar = colorbar('Location','eastoutside');
set(gca, 'FontSize', 24, 'FontName', 'Times');
% set(gca,'YTick',1:size(m_all, 2),'YTickLabel', m_all)
% set(gca,'XTick',1:size(eps_0_all, 2),'XTickLabel', eps_0_all)

fig_name = strcat('figures\bagging_single_tmatr_K_', num2str(K), '_m_', num2str(m));
saveas(h, strcat(fig_name, '.png'), 'png');
saveas(h, strcat(fig_name, '.eps'), 'psc2');


alpha = 0.001; % high penalization for redundant models, since K is big

h1=figure;
hold('on');
imagesc(t_matr > alpha);
set(gca, 'FontSize', 24, 'FontName', 'Times');

xlabel('$k$','FontSize', 24, 'Interpreter', 'latex');
ylabel('$l$','FontSize',24, 'Interpreter', 'latex');


axis 'tight' 'square'
bar = colorbar('Location','eastoutside');
set(gca, 'FontSize', 24, 'FontName', 'Times');
% set(gca,'YTick',1:size(m_all, 2),'YTickLabel', m_all)
% set(gca,'XTick',1:size(eps_0_all, 2),'XTickLabel', eps_0_all)

fig_name = strcat('figures\bagging_single_tmatr_K_', num2str(K), '_m_', num2str(m), '_geq_', num2str(alpha));
saveas(h1, strcat(fig_name, '.png'), 'png');
saveas(h1, strcat(fig_name, '.eps'), 'psc2');


% Several distinct models in bagging

m = 5000;
n = 5;
X = randn(m, n);
w_init = randn(n, 2);

y = generate_mixture_logistic(X, w_init, [0.5, 0.5]');


K = 1000;
sample_weights = zeros(m, K); % objects' weights
for k=1:K
   cur_sample_idx = randsample(m, m, true);
   occurences = hist(cur_sample_idx, 1:m);
   sample_weights(:, k) = occurences';
end

w = zeros(n, K);
hessian = cell(K, 1);

for k=1:K
    k
    [A, cur_w, cur_hessian, L] = maximize_evidence_single_logistic_laplace(X, y, eye(n), zeros(n, 1), 1e-5, 1, 20, 1e-10, sample_weights(:, k));
    cur_w(diag(A) >= 1e5) = 0;
    w(:, k) = cur_w;
    hessian{k} = cur_hessian;
end


t_matr = ones(K, K);

for k=1:K
    k
    for l=k:K
        t_matr(k, l) = get_significance_level_no_intersect(w(:, k), hessian{k}, w(:, l), hessian{l});
        t_matr(l, k) = t_matr(k, l);
    end
end

h=figure;
hold('on');
imagesc(t_matr);
set(gca, 'FontSize', 24, 'FontName', 'Times');

xlabel('$k$','FontSize', 24, 'Interpreter', 'latex');
ylabel('$l$','FontSize',24, 'Interpreter', 'latex');


axis 'tight' 'square'
bar = colorbar('Location','eastoutside');
set(gca, 'FontSize', 24, 'FontName', 'Times');
% set(gca,'YTick',1:size(m_all, 2),'YTickLabel', m_all)
% set(gca,'XTick',1:size(eps_0_all, 2),'XTickLabel', eps_0_all)

fig_name = strcat('figures\bagging_single_tmatr_K_', num2str(K), '_m_', num2str(m));
saveas(h, strcat(fig_name, '.png'), 'png');
saveas(h, strcat(fig_name, '.eps'), 'psc2');


alpha = 0.001; % high penalization for redundant models, since K is big

h1=figure;
hold('on');
imagesc(t_matr > alpha);
set(gca, 'FontSize', 24, 'FontName', 'Times');

xlabel('$k$','FontSize', 24, 'Interpreter', 'latex');
ylabel('$l$','FontSize',24, 'Interpreter', 'latex');


axis 'tight' 'square'
bar = colorbar('Location','eastoutside');
set(gca, 'FontSize', 24, 'FontName', 'Times');
% set(gca,'YTick',1:size(m_all, 2),'YTickLabel', m_all)
% set(gca,'XTick',1:size(eps_0_all, 2),'XTickLabel', eps_0_all)

fig_name = strcat('figures\bagging_single_tmatr_K_', num2str(K), '_m_', num2str(m), '_geq_', num2str(alpha));
saveas(h1, strcat(fig_name, '.png'), 'png');
saveas(h1, strcat(fig_name, '.eps'), 'psc2');