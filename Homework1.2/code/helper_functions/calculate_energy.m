function [E, W, e] = calculate_energy(camera_params, R, t, M, m)
% Projecting 3D points M to image -- world_points is M
image_points = worldToImage(camera_params, R, t, M);
% Claculate the reprojection error betweeen projected points and image
reprojection_err = image_points - m';
% converting into 2N by 1 vector --- from sheet e = [e 1 e 2 . . . e 2N ] T
e = [reprojection_err(:, 1)';reprojection_err(:, 2)'];
%disp(size(e));
e = e(:);
%disp(size(e));
% Homework sheet
MAD = median(abs(e)); % TODO: check if abs(e) is necessary
sigma = 1.48257968*MAD; % compute scale
c = 4.685;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fixed %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fixed %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W = TurkeyBiSquareWeights(e/sigma, c);
% Slide 47
E = sum(TurkeyBiSquareMEstimator(e, c));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Moved into functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Fall back to this in case of issue %%%%%%%%%%%%%%%%%
% Sheet
% abs_e = abs(e);
% sigma_e = e/sigma;
% sigma_e = abs(sigma_e);
% turkeyBiSqW_condition_true = find(sigma_e<c);
% turkeyBiSqW_condition_false = find(sigma_e>=c); % find(sigma_e>=c);
% weights(turkeyBiSqW_condition_true) = (1 - e(turkeyBiSqW_condition_true).^2/(c^2)).^2;
% weights(turkeyBiSqW_condition_false) = 0;
% W = diag(weights);
% % Sheet
% turkeyBiSqMEstimator_condition_true = find(abs_e<=c);
% turkeyBiSqMEstimator_condition_false = find(abs_e>c); % find(e>c);
% RHO(turkeyBiSqMEstimator_condition_true) = (c^2)/6.*(1 - (1 - (e(turkeyBiSqMEstimator_condition_true)./c).^2).^3);
% RHO(turkeyBiSqMEstimator_condition_false) = (c^2)/6;
% E = sum(RHO);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end