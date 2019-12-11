function [W] = TurkeyBiSquareWeights(e, c)
e = abs(e);
turkeyBiSqW_condition_true = find(e<c);
turkeyBiSqW_condition_false = find(e>=c); % find(sigma_e>=c);
weights(turkeyBiSqW_condition_true) = (1 - e(turkeyBiSqW_condition_true).^2/(c^2)).^2;
weights(turkeyBiSqW_condition_false) = 0;
W = diag(weights);
end
