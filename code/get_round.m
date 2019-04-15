function auc_text = get_round(auc, num_digits)
    auc_round = round(10^num_digits * auc) * 0.1^num_digits;
    auc_text = num2str(auc_round);
return