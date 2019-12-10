function [w] = TurkeyBiSquareWeights(e, c)
e = abs(e);

if e < c
    w = (1 - e.^2/(c^2)).^2;
else
    w = 0;
end
end
