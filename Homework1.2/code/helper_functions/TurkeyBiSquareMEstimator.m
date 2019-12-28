function [RHO] = TurkeyBiSquareMEstimator(e, c)
e = abs(e);
turkeyBiSqMEstimator_condition_true = find(e<=c);
turkeyBiSqMEstimator_condition_false = find(e>c); % find(e>c);
RHO(turkeyBiSqMEstimator_condition_true) = (c^2)/6.*(1 - (1 - (e(turkeyBiSqMEstimator_condition_true)./c).^2).^3);
RHO(turkeyBiSqMEstimator_condition_false) = (c^2)/6;
end