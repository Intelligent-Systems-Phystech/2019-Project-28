% m = 5000;
% n = 2;
% X = randn(m, n);
% w = [1 1; 1 -1];
% 
% y = generate_mixture_logistic(X, w, [0.6, 0.4]');
% 
% alpha = 0.0001;
% K = 20;
% [w, pi, hessian, A, L] = learn_optimal_mixture_logistic(X, y, K, alpha);
% 
% h=figure;
% hold('on');
% 
% plot(L, 'LineWidth', 2);
% 
% %legend('ROC-curve','sample set','random guessing');
% set(gca, 'FontSize', 24, 'FontName', 'Times');
% %legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
% %set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
% axis('tight');
% %axis([-2.5, 2.5, 0, 6.5])
% 
% xlabel('Iteration','FontSize',24, 'Interpreter', 'latex');
% ylabel('$p(\mathbf{y}|\mathbf{X},\:\mathbf{A})$','FontSize',24, 'Interpreter', 'latex');
% 
% fig_name = strcat('figures\mixture_multiextreme_K_', num2str(K), '_alpha_',...
%     num2str(alpha), '_m_', num2str(m));
% saveas(h, strcat(fig_name, '.png'), 'png');
% saveas(h, strcat(fig_name, '.eps'), 'psc2');


% Multiple extremes for learning multimodel
m = 1000;
n = 2;
X = randn(m, n);
w_init = [1 1; 1 -1];
pi_init = [0.6, 0.4]';

y = generate_mixture_logistic(X, w_init, pi_init);
auc_init = test_mixture_logistic(X, y, pi_init, w_init)
sigma_init = ones(m, K) ./ (1 + exp(-(X * w_init) .* (y * ones(1, K))));
prob_init = sigma_init .* (ones(m, 1) * pi_init');
sum_prob_init = prob_init * ones(K, 1);
prob_init = prob_init ./ (sum_prob_init * ones(1, K));

L_init = count_mixture_learn_L(X, y, A, alpha, pi_init, w_init, prob_init)

alpha = 1;
K = 2;

A = cell(K, 1);
for k=1:K
   A{k} = zeros(n, n); 
end
[w, pi, hessian, L, pi_evolution, w_evolution] = learn_mixture_logistic(X, y, A, alpha);
auc_new = test_mixture_logistic(X, y, pi, w)


max_pi_evolution = zeros(size(pi_evolution, 1), size(pi_evolution, 2));
iter_passed = size(pi_evolution, 1) * ones(size(pi_evolution, 2), 1);
for iter=1:size(pi_evolution, 1) 
    for index=1:size(pi_evolution, 2)
        cur_max = nanmax(cell2mat(pi_evolution(iter, index)));
        if (size(cur_max, 1))
            max_pi_evolution(iter, index) = cur_max;
        else
            max_pi_evolution(iter, index) = max_pi_evolution(iter - 1, index);
            iter_passed(index, 1) = min(iter_passed(index, 1), iter - 1);
        end
    end
end


h2=figure;
hold('on');

plot(max_pi_evolution, 'LineWidth', 2);
for iter=1:size(iter_passed, 1)
    iter, cell2mat(pi_evolution(iter_passed(iter, 1), iter))
    plot(iter_passed(iter, 1), max_pi_evolution(iter_passed(iter, 1), iter), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
end

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

L_denaned = L;

for index=2:size(L, 1)
   for j=1:size(L, 2)
      if (isnan(L(index, j)))
          L_denaned(index, j) = L_denaned(index - 1, j);
      end
   end
end


h1=figure;
hold('on');

plot(L_denaned(2:size(L_denaned, 1),:) , 'LineWidth', 2);

%legend('ROC-curve','sample set','random guessing');
set(gca, 'FontSize', 24, 'FontName', 'Times');
%legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
%set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
axis('tight');
%axis([-2.5, 2.5, 0, 6.5])

xlabel('Iteration','FontSize',24, 'Interpreter', 'latex');
ylabel('$\log p(\mathbf{y},\:\pi,\:\mathbf{w}_1,\:\ldots,\:\mathbf{w}_K|\mathbf{X})$','FontSize',24, 'Interpreter', 'latex');

fig_name = strcat('figures\learn_mixture_multiextreme_K_', num2str(K), '_alpha_',...
    num2str(alpha), '_m_', num2str(m));
saveas(h1, strcat(fig_name, '.png'), 'png');
saveas(h1, strcat(fig_name, '.eps'), 'psc2');



% Showing possible inadequacy of a mixture
% (it is always possible to build adequate mixture (just use 1 model), but
% the quality might suffer from this

% m = 5000;
% n = 15;
% K = 10;
% X = randn(m, n);
% 
% w_0 = randn(n, 1);
% rho = 0.99;
% 
% w_init = rho * w_0 * ones(1, K) + sqrt(1 - rho^2) * randn(n, K);
% y = generate_mixture_logistic(X, w_init, 1 / K * ones(K, 1));
% 
% A = cell(K, 1);
% for k=1:K
%    A{k} = zeros(n, n); 
% end
% [w, pi, hessian, L, pi_evolution, w_evolution] = learn_mixture_logistic(X, y, A, 1.);
% 
% t_matr = zeros(K, K);
% for i=1:(K-1)
%     for j=(i+1):K
%         t_matr(i, j) = get_significance_level_no_intersect(w(:, i), hessian{i}, w(:, j), hessian{j});
%         t_matr(j, i) = t_matr(i, j);
%     end
% end





