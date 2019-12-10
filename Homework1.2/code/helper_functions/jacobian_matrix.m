function [jacobian] = jacobian_matrix(R, t, M, camera_params)
% Total number of points
num_points = size(M, 1);
% Placing computed Jacobian matrix here
jacobian = zeros(2*num_points, 6);

% Computing the projection matrix
P = [R ; t]*camera_params.IntrinsicMatrix;
% Convert to Homogeneous Cordinate System
M_homogeneous = [M ones(num_points, 1)];
% Projecting 3D to 2D
image_points = M_homogeneous*P;

% SO(3) to so(3) 
r = rotationMatrixToVector(R);
r_skew_matrix = SkewSymmetricMatrix(r);

% derivative of a 3-D rotation in exponential coordinates
e = eye(3); % Standard Basis

dR_dr1 = ((r(1)*r_skew_matrix + SkewSymmetricMatrix(cross(r', (eye(3) - R')*e(1,:)')))./norm(r)^2)*R';
dR_dr2 = ((r(2)*r_skew_matrix + SkewSymmetricMatrix(cross(r', (eye(3) - R')*e(2,:)')))./norm(r)^2)*R';
dR_dr3 = ((r(3)*r_skew_matrix + SkewSymmetricMatrix(cross(r', (eye(3) - R')*e(3,:)')))./norm(r)^2)*R';

for i = 1:num_points
    % Slide 36
    dm_tilde__dMcam = camera_params.IntrinsicMatrix;
    % Slide 35
    U = image_points(i, 1);
    V = image_points(i, 2);
    W = image_points(i, 3);
    dm__dm_tilde = [1/W,   0,  -U/W^2;
                    0,    1/W, -V/W^2];
    % Slide 37
    dMcam_dp = [zeros(3) eye(3)];    
    dMcam_dp(:, 1) = dR_dr1*M(i, :)';
    dMcam_dp(:, 2) = dR_dr2*M(i, :)';
    dMcam_dp(:, 3) = dR_dr3*M(i, :)';
    
    % Slide 34
    jacobian((2*i-1):(2*i), :) = dm__dm_tilde*dm_tilde__dMcam'*dMcam_dp;
end
end