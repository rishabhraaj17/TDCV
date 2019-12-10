function [rho] = TurkeyBiSquareMEstimator(e, c)
e = abs(e);
if e <= c
    rho = (c^2)/6.*(1 - (1 - (e./c).^2).^3);
else
    rho = (c^2)/6;
end
end