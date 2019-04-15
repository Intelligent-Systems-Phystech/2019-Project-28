function [AUC, AUC_joint] = test_multilevel_logistic(X, y, w, idx)
% w is a cell array
% idx indicates split between models
    K = size(w, 1);
    AUC = nan * zeros(1, K);
    joint_prob = zeros(size(X, 1), 1);
    for k=1:K
        cur_idx = (idx == k);
        cur_m = sum(cur_idx);
        if (cur_m < 2)
           continue
        end
        %cur_m, k, size(cur_idx)
        prob = ones(cur_m, 1) ./ (1 + exp(-X(cur_idx, :) * w{k}));
        joint_prob(cur_idx) = prob;
        try
            [~, ~, ~, AUC(1, k)] = perfcurve(y(cur_idx), prob, 1);
        catch
            AUC(1, k) = nan;
        end
    end
    try
       [~, ~, ~, AUC_joint] = perfcurve(y, joint_prob, 1);
    catch
       AUC_joint = nan; 
    end
return