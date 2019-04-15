m = 5000;
n = 2;
X = randn(m, n);
w = [1 1; 1 -1];

y = generate_mixture_logistic(X, w, [0.6, 0.4]');

alpha = 0.0001;
K = 20;
[w, pi, hessian, A, L] = learn_optimal_mixture_logistic(X, y, K, alpha);

h=figure;
hold('on');

plot(L, 'LineWidth', 2);

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
%legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
%set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
axis('tight');
%axis([-2.5, 2.5, 0, 6.5])

xlabel('Iteration','FontSize',24, 'Interpreter', 'latex');
ylabel('$p(\mathbf{y}|\mathbf{X},\:\mathbf{A})$','FontSize',24, 'Interpreter', 'latex');

fig_name = strcat('figures\mixture_multiextreme_K_', num2str(K), '_alpha_',...
    num2str(alpha), '_m_', num2str(m));
saveas(h, strcat(fig_name, '.png'), 'png');
saveas(h, strcat(fig_name, '.eps'), 'psc2');


% Multiple extremes for learning multimodel

A = cell(K, 1);
for k=1:K
   A{k} = zeros(n, n); 
end
[w, pi, hessian, L, pi_evolution, w_evolution] = learn_mixture_logistic(X, y, A, alpha);

h1=figure;
hold('on');

plot(L, 'LineWidth', 2);

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
%legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
%set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
axis('tight');
%axis([-2.5, 2.5, 0, 6.5])

xlabel('Iteration','FontSize',24, 'Interpreter', 'latex');
ylabel('$p(\mathbf{y},\:\pi,\:\mathbf{w}_1,\:\ldots,\:\mathbf{w}_K|\mathbf{X})$','FontSize',24, 'Interpreter', 'latex');

fig_name = strcat('figures\learn_mixture_multiextreme_K_', num2str(K), '_alpha_',...
    num2str(alpha), '_m_', num2str(m));
saveas(h1, strcat(fig_name, '.png'), 'png');
saveas(h1, strcat(fig_name, '.eps'), 'psc2');

max_pi_evolution = zeros(size(pi_evolution, 1), size(pi_evolution, 2));
for iter=1:size(pi_evolution, 1) 
    for index=1:size(pi_evolution, 2)
        cur_max = nanmax(cell2mat(pi_evolution(iter, index)));
        if (size(cur_max, 1))
            max_pi_evolution(iter, index) = cur_max;
        else
            max_pi_evolution(iter, index) = max_pi_evolution(iter - 1, index);
        end
    end
end


h2=figure;
hold('on');

plot(max_pi_evolution, 'LineWidth', 2);

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
%legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
%set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
axis('tight');
%axis([-2.5, 2.5, 0, 6.5])

xlabel('Iteration','FontSize',24, 'Interpreter', 'latex');
ylabel('$\max_{k} \pi_k$','FontSize',24, 'Interpreter', 'latex');

fig_name = strcat('figures\learn_mixture_multiextreme_max_pi_evol_K_', num2str(K), '_alpha_',...
    num2str(alpha), '_m_', num2str(m));
saveas(h2, strcat(fig_name, '.png'), 'png');
saveas(h2, strcat(fig_name, '.eps'), 'psc2');


% Showing possible inadequacy of a mixture
% (it is always possible to build adequate mixture (just use 1 model), but
% the quality might suffer from this

m = 5000;
n = 2;
K = 10;
X = randn(m, n);

w_0 = randn(n, 1);
rho = 0.;

% w_init = rho * w_0 * ones(1, K) + sqrt(1 - rho^2) * randn(n, K);
w_init = (2 * 3.14159 / K) * linspace(0, K - 1, K);
w_init = [cos(w_init); sin(w_init)];
%weights = 5 * abs(randn(1, K));
weights = 5 * sqrt(rand(1, K));
w_init = w_init .* [weights; weights];

min_dist = 1e10;
for i=1:(K-1)
    for j=(i+1):K
        min_dist = min(min_dist, norm(w_init(:, i) - w_init(:, j)));
        %min_dist = min(min_dist, norm(w_init(:, i) + w_init(:, j)));
    end
end

y = generate_mixture_logistic(X, w_init, 1 / K * ones(K, 1));

A = cell(K, 1);
for k=1:K
   A{k} = zeros(n, n); 
end
[w, pi, hessian, L, pi_evolution, w_evolution] = learn_mixture_logistic(X, y, A, 1.);

min_dist_w = 1e10;
for i=1:(K-1)
    for j=(i+1):K
        min_dist_w = min(min_dist_w, norm(w(:, i) - w(:, j)));
    end
end

t_matr = ones(K, K);
for i=1:(K-1)
    for j=(i+1):K
        t_matr(i, j) = get_significance_level_no_intersect(w(:, i), hessian{i}, w(:, j), hessian{j});
        t_matr(j, i) = t_matr(i, j);
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

fig_name = strcat('figures\mixture_inadequacy_tmatr_K_', num2str(K), '_m_', num2str(m));
saveas(h, strcat(fig_name, '.png'), 'png');
saveas(h, strcat(fig_name, '.eps'), 'psc2');
    
h1=figure;
hold('on');
imagesc(t_matr >= 0.05);
set(gca, 'FontSize', 24, 'FontName', 'Times');

xlabel('$k$','FontSize', 24, 'Interpreter', 'latex');
ylabel('$l$','FontSize',24, 'Interpreter', 'latex');


axis 'tight' 'square'
bar = colorbar('Location','eastoutside');
set(gca, 'FontSize', 24, 'FontName', 'Times');
% set(gca,'YTick',1:size(m_all, 2),'YTickLabel', m_all)
% set(gca,'XTick',1:size(eps_0_all, 2),'XTickLabel', eps_0_all)

fig_name = strcat('figures\mixture_inadequacy_tmatr_similar_0.05_K_', num2str(K), '_m_', num2str(m));
saveas(h1, strcat(fig_name, '.png'), 'png');
saveas(h1, strcat(fig_name, '.eps'), 'psc2');



