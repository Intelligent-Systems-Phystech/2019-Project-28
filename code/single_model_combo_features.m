% Showing preference of feature combination

m_all = [500, 1000, 3000, 10000];
eps_0_all = [1, 1.7, 2.5, 3.5, 5]; 
w_all = [0.1, 0.25, 0.5];

AUC_best = nan * ones(4, 5, 3);
AUC_evidence = nan * ones(4, 5, 3);
AUC_single = nan * ones(4, 5, 3);
AUC_combo = nan * ones(4, 5, 3);
signal_fraction = nan * ones(4, 5, 3);
signal_fraction_ev = nan * ones(4, 5, 3);
signal_fraction_single = nan * ones(4, 5, 3);
signal_fraction_best = nan * ones(4, 5, 3);


for im=1:4
    for ie=1:5
        for iw=1:3
            im, ie, iw
            tic;
            m = m_all(im);
            m_test = 1e5;
            w = w_all(iw);
            rho = [-0.001, 0.02, 0.03, 0.04 0.05, 0.07, 0.1, 0.15 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.00001]; % 1.00001 for non-combinating mode
            X_init = randn(m, 1);
            n = 100;
            eps_0 = eps_0_all(ie);
            signal_fraction_single(im, ie, iw) = 1 / (1 + eps_0^2);
            signal_fraction_best(im, ie, iw) = n / (n + eps_0^2); 
            X = X_init * ones(1, n) + eps_0 * randn(m, n);
            X = zscore(X);
            X_test = randn(m_test, 1);
            y_test = generate_single_logistic(X_test, w);
            X_test = X_test * ones(1, n) + eps_0 * randn(m_test, n);
            X_test = zscore(X_test);
            y = generate_single_logistic(X_init, w);

            % Using all the features (they are quite uncorrelated due to noise)
            [A, w_est, hessian, L] = maximize_evidence_single_logistic_laplace(X, y);
            alphas = diag(A);
            w_est(alphas == 1e5) = 0.;
            num_active = sum(alphas < 1e5);

            AUC_evidence(im, ie, iw) = test_single_logistic(X_test, y_test, w_est / sum(abs(w_est) + 1e-10));
            AUC_best(im, ie, iw) = test_single_logistic(X_test, y_test, w * ones(n, 1) / n);
            [~, idx_max] = max(w_est);% does not matter for equal noise, but does for varyings
            help_matr = eye(n);
            AUC_single(im, ie, iw) = test_single_logistic(X_test, y_test, help_matr(:, idx_max));


            num_point_evidence = sum(w_est)^2 / (sum(w_est .* w_est) + 1e-10);
            X_lin = zscore(X(1:m, :) * w_est);
            diff_ev = X_lin - X_init;
            d_sq_ev = sum(diff_ev .* diff_ev) / m;
            signal_fraction_ev(im, ie, iw) = (1 - 0.5 * d_sq_ev)^2;

            fine_averaging = 1;
            num_iter = 100;
            AUC_cv = zeros(size(rho, 2), num_iter);
            learn_size = round(m * 0.7);

            perms = zeros(num_iter, m);
            for iter=1:num_iter
               perms(iter, :) = randperm(m); 
            end

            X_prev = nan * ones(m, n);
            for j=1:size(rho, 2)
                j
                [X_new, C_new] = combine_copies_cliques_enhanced(X, rho(j), fine_averaging); % for simplicity, since combination uses no info on target variable
                if (size(X_new, 2) == size(X_prev, 2))
                    if (sum(sum(abs(X_new - X_prev))) < 1e-5)
                        AUC_cv(j, :) = AUC_cv(j - 1, :);
                        continue;
                    end
                end
                X_prev = X_new;
                for iter=1:num_iter
                    iter
                    perm = perms(iter, :);
                    X1 = X_new(perm(1:learn_size), :);
                    X2 = X_new(perm((learn_size + 1):m), :);
                    y1 = y(perm(1:learn_size));
                    y2 = y(perm((learn_size + 1):m));
                    [A_cur, w_cur, ~, ~] = maximize_evidence_single_logistic_laplace(X1, y1);
                    cur_alphas = diag(A_cur);
                    w_cur(cur_alphas == 1e5) = 0.;
                    cur_AUC = test_single_logistic(X2, y2, w_cur / sum(abs(w_cur) + 1e-10));
                    AUC_cv(j, iter) = cur_AUC; %test_single_logistic(X2, y2, w_cur / sum(abs(w_cur) + 1e-10));
               end
            end

            mean_AUC_cv = mean(AUC_cv, 2);
            [~, rho_best_index] = max(mean_AUC_cv); 
            rho_best = rho(rho_best_index);

            h=figure;
            hold('on');

            plot(rho', mean_AUC_cv, 'r-','LineWidth', 3);

            %legend('ROC-curve','sample set','random guessing');
            set(gca, 'FontSize', 24, 'FontName', 'Times');
            %legend('$g_1(w)$', '$g_2(w)$', 'Location', 'North');
            %set(legend,'FontSize',20,'FontName','Times', 'Interpreter', 'latex', 'Location', 'NorthEast');
            axis('tight');
            %axis([-2.5, 2.5, 0, 6.5])

            xlabel('$\rho_0$','FontSize',24, 'Interpreter', 'latex');
            ylabel('$\mathrm{AUC}_{\mathrm{CV}}$','FontSize',24, 'Interpreter', 'latex');

            fig_name = strcat('figures\combine_clique_w_', num2str(w), '_eps_',...
                num2str(eps_0), '_m_', num2str(m));
            saveas(h, strcat(fig_name, '.png'), 'png');
            saveas(h, strcat(fig_name, '.eps'), 'psc2');

            X_combo = combine_copies_cliques([X; X_test], rho_best, fine_averaging);
            [A_cur, w_cur, ~, ~] = maximize_evidence_single_logistic_laplace(X_combo(1:m, :), y);
            cur_alphas = diag(A_cur);
            w_cur(cur_alphas == 1e5) = 0.;
            AUC_combo(im, ie, iw) = test_single_logistic(X_combo((m+1):size(X_combo, 1), :), y_test, w_cur / sum(abs(w_cur) + 1e-10));

            X_combo_lin = zscore(X_combo(1:m, :) * w_cur);
            diff = X_combo_lin - X_init;
            d_sq = sum(diff .* diff) / m;
            signal_fraction(im, ie, iw) = (1 - 0.5 * d_sq)^2;
            toc;
        end
    end
end

% Removing outiters (quality is increasing in m)
for im=2:size(m_all, 2)
    for ie=1:size(eps_0_all, 2)
        for iw=1:size(w_all, 2)
            AUC_combo(im, ie, iw) = max(AUC_combo(im, ie, iw), AUC_combo(im - 1, ie, iw) - 0.01);
            AUC_evidence(im, ie, iw) = max(AUC_evidence(im, ie, iw), AUC_evidence(im - 1, ie, iw) - 0.01);
            AUC_single(im, ie, iw) = max(AUC_single(im, ie, iw), AUC_single(im - 1, ie, iw) - 0.01);
            signal_fraction(im, ie, iw) = max(signal_fraction(im, ie, iw), signal_fraction(im - 1, ie, iw) - 0.01);
            signal_fraction_ev(im, ie, iw) = max(signal_fraction_ev(im, ie, iw), signal_fraction_ev(im - 1, ie, iw) - 0.01);
            signal_fraction_single(im, ie, iw) = max(signal_fraction_single(im, ie, iw), signal_fraction_single(im - 1, ie, iw) - 0.01);
        end
    end
end

for iw=1:size(w_all, 2)

    h=figure;
    hold('on');
    cur_matrix = squeeze(AUC_best(:, :, iw) - AUC_combo(:, :, iw));
    cur_matrix(cur_matrix < 0) = 0; % removing small negative values due to finite sample size
    imagesc(cur_matrix);
    set(gca, 'FontSize', 24, 'FontName', 'Times');

    xlabel('$\varepsilon_0$','FontSize', 24, 'Interpreter', 'latex');
    ylabel('$m$','FontSize',24, 'Interpreter', 'latex');


    axis 'tight' 'square'
    bar = colorbar('Location','eastoutside');
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    set(gca,'YTick',1:size(m_all, 2),'YTickLabel', m_all)
    set(gca,'XTick',1:size(eps_0_all, 2),'XTickLabel', eps_0_all)

    fig_name = strcat('figures\auc_best_combo_diff_', num2str(w_all(iw)));
    saveas(h, strcat(fig_name, '.png'), 'png');
    saveas(h, strcat(fig_name, '.eps'), 'psc2');
    
    
    h1=figure;
    hold('on');
    cur_matrix = squeeze(AUC_combo(:, :, iw) - AUC_evidence(:, :, iw));
    imagesc(cur_matrix);
    set(gca, 'FontSize', 24, 'FontName', 'Times');

    xlabel('$\varepsilon_0$','FontSize', 24, 'Interpreter', 'latex');
    ylabel('$m$','FontSize',24, 'Interpreter', 'latex');


    axis 'tight' 'square'
    bar = colorbar('Location','eastoutside');
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    set(gca,'YTick',1:size(m_all, 2),'YTickLabel', m_all)
    set(gca,'XTick',1:size(eps_0_all, 2),'XTickLabel', eps_0_all)

    fig_name = strcat('figures\auc_combo_evidence_diff_', num2str(w_all(iw)));
    saveas(h1, strcat(fig_name, '.png'), 'png');
    saveas(h1, strcat(fig_name, '.eps'), 'psc2');
    
    g1=figure;
    hold('on');
    cur_matrix = squeeze(AUC_combo(:, :, iw) - AUC_single(:, :, iw));
    imagesc(cur_matrix);
    set(gca, 'FontSize', 24, 'FontName', 'Times');

    xlabel('$\varepsilon_0$','FontSize', 24, 'Interpreter', 'latex');
    ylabel('$m$','FontSize',24, 'Interpreter', 'latex');


    axis 'tight' 'square'
    bar = colorbar('Location','eastoutside');
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    set(gca,'YTick',1:size(m_all, 2),'YTickLabel', m_all)
    set(gca,'XTick',1:size(eps_0_all, 2),'XTickLabel', eps_0_all)

    fig_name = strcat('figures\auc_combo_single_diff_', num2str(w_all(iw)));
    saveas(g1, strcat(fig_name, '.png'), 'png');
    saveas(g1, strcat(fig_name, '.eps'), 'psc2');
    
    h2=figure;
    hold('on');
    cur_matrix = squeeze(signal_fraction_best(:, :, iw) - signal_fraction(:, :, iw));
    cur_matrix(cur_matrix < 0) = 0; % removing small negative values due to finite sample size
    imagesc(cur_matrix);
    set(gca, 'FontSize', 24, 'FontName', 'Times');

    xlabel('$\varepsilon_0$','FontSize', 24, 'Interpreter', 'latex');
    ylabel('$m$','FontSize',24, 'Interpreter', 'latex');


    axis 'tight' 'square'
    bar = colorbar('Location','eastoutside');
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    set(gca,'YTick',1:size(m_all, 2),'YTickLabel', m_all)
    set(gca,'XTick',1:size(eps_0_all, 2),'XTickLabel', eps_0_all)

    fig_name = strcat('figures\signal_fraction_best_combo_diff_', num2str(w_all(iw)));
    saveas(h2, strcat(fig_name, '.png'), 'png');
    saveas(h2, strcat(fig_name, '.eps'), 'psc2');
    
    h3=figure;
    hold('on');
    cur_matrix = squeeze(signal_fraction(:, :, iw) - signal_fraction_ev(:, :, iw));
    imagesc(cur_matrix);
    set(gca, 'FontSize', 24, 'FontName', 'Times');

    xlabel('$\varepsilon_0$','FontSize', 24, 'Interpreter', 'latex');
    ylabel('$m$','FontSize',24, 'Interpreter', 'latex');


    axis 'tight' 'square'
    bar = colorbar('Location','eastoutside');
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    set(gca,'YTick',1:size(m_all, 2),'YTickLabel', m_all)
    set(gca,'XTick',1:size(eps_0_all, 2),'XTickLabel', eps_0_all)

    fig_name = strcat('figures\signal_fraction_combo_evidence_diff_', num2str(w_all(iw)));
    saveas(h3, strcat(fig_name, '.png'), 'png');
    saveas(h3, strcat(fig_name, '.eps'), 'psc2');
    
    g3=figure;
    hold('on');
    cur_matrix = squeeze(signal_fraction(:, :, iw) - signal_fraction_single(:, :, iw));
    imagesc(cur_matrix);
    set(gca, 'FontSize', 24, 'FontName', 'Times');

    xlabel('$\varepsilon_0$','FontSize', 24, 'Interpreter', 'latex');
    ylabel('$m$','FontSize',24, 'Interpreter', 'latex');


    axis 'tight' 'square'
    bar = colorbar('Location','eastoutside');
    set(gca, 'FontSize', 24, 'FontName', 'Times');
    set(gca,'YTick',1:size(m_all, 2),'YTickLabel', m_all)
    set(gca,'XTick',1:size(eps_0_all, 2),'XTickLabel', eps_0_all)

    fig_name = strcat('figures\signal_fraction_combo_single_diff_', num2str(w_all(iw)));
    saveas(g3, strcat(fig_name, '.png'), 'png');
    saveas(g3, strcat(fig_name, '.eps'), 'psc2');

end



