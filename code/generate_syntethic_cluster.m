function [X, y, idx, AUC_adequate_mean, AUC_adequate_min, AUC_adequate_std, AUC_multilevel_mean, AUC_multilevel_min, AUC_multilevel_std, num_groups] = generate_syntethic_cluster(m_init, K, size_inequality, scale, to_calc, to_plot)
    if (nargin < 6)
       to_plot = 0; 
    end
    if (nargin < 5)
        to_calc = 0;
    end
    if (nargin < 4)
       scale = 0.1; 
    end
    if (nargin < 3)
       size_inequality = 1; 
    end
    sizes = round(m_init / K) * ones(K, 1);
    if (size_inequality > 1)
        delta = power(size_inequality, 1/(K - 1));
        sum_weight = (power(delta, K) - 1) / (delta - 1);
        idx = randperm(K);
        sizes(idx(1), 1) = (m_init + 0.) / (sum_weight + 0.);
        for k=2:K
            sizes(idx(k), 1) = sizes(idx(k-1), 1) * delta;
        end
        for k=1:K
            sizes(k, 1) = max(10, round(sizes(k, 1))); % increasing too small sizes
        end
    end
    w = scale * [1, -1]';
    m = sum(sizes);
    n = 2;
    X = randn(m, 2);
    idx = zeros(m, 1);
    sparse_coeff = 2.5;
    shifts = linspace(-sparse_coeff * (K - 1), sparse_coeff * (K - 1),K);
    
    start = 1;
    for k=1:K
       X(start:(start + sizes(k, 1) - 1), :) = X(start:(start + sizes(k, 1) - 1), :) + shifts(k); 
       idx(start:(start + sizes(k, 1) - 1), 1) = k;
       start = start + sizes(k, 1);
    end
    prob = ones(m, 1) ./ (1 + exp(-X * w));
    y = 2 * (rand(m, 1) < prob) - 1;
    
    if (to_calc == 1)
        if (to_plot == 1)
            h=figure;
            hold('on');

            plot(X(:, 1), X(:, 2), 'r.','MarkerSize', 10);
            phi = linspace(0, 6.3, 1000);
            for k=1:K
               plot(cos(phi) * sqrt(2) * sparse_coeff + shifts(k), sin(phi) * sqrt(2) * sparse_coeff + shifts(k), 'b-', 'LineWidth', 2);
            end

            %legend('ROC-curve','sample set','random guessing');
            set(gca, 'FontSize', 24, 'FontName', 'Times');
            %legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
            %set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
            axis('square');
            %axis([-2.5, 2.5, 0, 6.5])

            xlabel('$x_1$','FontSize',24, 'Interpreter', 'latex');
            ylabel('$x_2$','FontSize',24, 'Interpreter', 'latex');

            fig_name = strcat('figures\cluster_multilevel_with_circles_m_', num2str(m_init), '_K_',...
                num2str(K), '_ineq_', num2str(size_inequality));
            saveas(h, strcat(fig_name, '.png'), 'png');
            saveas(h, strcat(fig_name, '.eps'), 'psc2');

            h1=figure;
            hold('on');

            plot(X(:, 1), X(:, 2), 'r.','MarkerSize', 10);

            %legend('ROC-curve','sample set','random guessing');
            set(gca, 'FontSize', 24, 'FontName', 'Times');
            %legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
            %set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
            axis('square');
            %axis([-2.5, 2.5, 0, 6.5])

            xlabel('$x_1$','FontSize',24, 'Interpreter', 'latex');
            ylabel('$x_2$','FontSize',24, 'Interpreter', 'latex');

            fig_name = strcat('figures\cluster_multilevel_no_circles_m_', num2str(m_init), '_K_',...
                num2str(K), '_ineq_', num2str(size_inequality));
            saveas(h1, strcat(fig_name, '.png'), 'png');
            saveas(h1, strcat(fig_name, '.eps'), 'psc2');

            idx_pos = (y == 1);
            idx_neg = (y == -1);

            h2=figure;
            hold('on');

            plot(X(idx_pos, 1), X(idx_pos, 2), 'r.','MarkerSize', 10);
            plot(X(idx_neg, 1), X(idx_neg, 2), 'b.','MarkerSize', 10);
            start_x = min(X(:, 1));
            end_x = max(X(:, 1));
            plot([start_x, end_x], [start_x, end_x], 'k-', 'LineWidth', 2);

            %legend('ROC-curve','sample set','random guessing');
            set(gca, 'FontSize', 24, 'FontName', 'Times');
            %legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
            %set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
            axis('square');
            %axis([-2.5, 2.5, 0, 6.5])

            xlabel('$x_1$','FontSize',24, 'Interpreter', 'latex');
            ylabel('$x_2$','FontSize',24, 'Interpreter', 'latex');

            fig_name = strcat('figures\cluster_multilevel_ideal_m_', num2str(m_init), '_K_',...
                num2str(K), '_ineq_', num2str(size_inequality));
            saveas(h2, strcat(fig_name, '.png'), 'png');
            saveas(h2, strcat(fig_name, '.eps'), 'psc2');
        end
        
        A = cell(K, 1);
        for k=1:K
           A{k} = zeros(n, n); 
        end
              
        [w_single, ~] = learn_single_logistic(X, y, zeros(n, n));
        [idx_new, mapping_new] = get_adequate_multilevel_model(X, y, A, idx, 0.05);
        if (max(idx_new) == 1)
           w_adequate = cell(1, 1);
           w_adequate{1} = w_single;  % all is put to single model
        else
           [w_adequate, ~] = learn_multilevel_logistic(X, y, A(1:max(idx_new), 1), idx_new);
        end
        
        num_groups = max(idx_new);
        
        [w_multilevel, ~] = learn_multilevel_logistic(X, y, A, idx);
        % estimating AUC
        X_new = randn(1000, n);
        %AUC_single = zeros(K, 1);
        AUC_multilevel = zeros(K, 1);
        AUC_adequate = zeros(K, 1);
        
        for k=1:K
           X_cur = X_new + shifts(k);
           y_cur = generate_single_logistic(X_cur, w);
           AUC_multilevel(k, 1) = test_single_logistic(X_cur, y_cur, w_multilevel{k});
           %AUC_single(k, 1) = test_single_logistic(X_cur, y_cur, w_single);
           AUC_adequate(k, 1) = test_single_logistic(X_cur, y_cur, w_adequate{mapping_new(k)});
        end
%         AUC_single_mean = mean(AUC_single);
%         AUC_single_min = min(AUC_single);
%         AUC_single_std = std(AUC_single);
        AUC_adequate_mean = mean(AUC_adequate);
        AUC_adequate_min = min(AUC_adequate);
        AUC_adequate_std = std(AUC_adequate);
        
        AUC_multilevel_mean = mean(AUC_multilevel);
        AUC_multilevel_min = min(AUC_multilevel);
        AUC_multilevel_std = std(AUC_multilevel);
        
    end


return