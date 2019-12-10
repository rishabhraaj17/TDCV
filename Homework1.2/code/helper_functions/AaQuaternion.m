function[quaternion] = AaQuaternion(R, varargin)
% Using the rotation matrix 'R'
% Do the Logarithm map: Inverse Mapping
% Then do the Quaternion with angle axis parameterisation
% TODO: check for the Singularity case if case arises.
% Default calculations will be in Radians
radian = true;
if ~isempty(varargin)
    options = varargin{1};
    if (isfield(options, 'radian'))
        radian = options.radian; 
    end
else
    radian = true;
end
eps = 1e-10;
if radian == true
    % disp("Angle in Radian");
    theta = acos(0.5*(trace(R) - 1));

    w = (0.5*sin(theta))*[R(3,2) - R(2,3);
                          R(1,3) - R(3,1); 
                          R(2,1) - R(1,2)];
    w_norm = norm(w);
    if w_norm == 0
        w_a = w/(w_norm + eps);
    else
        w_a = w/w_norm;
    end
    

    quaternion = [cos(w_norm/2), sin(w_norm/2)*w_a'];
else
    % disp("Angle in Degree");
    theta = acosd(0.5*(trace(R) - 1));

    w = (0.5*sind(theta))*[R(3,2) - R(2,3);
                           R(1,3) - R(3,1); 
                           R(2,1) - R(1,2)];
    w_norm = norm(w);
    if w_norm == 0
        w_a = w/(w_norm + eps);
    else
        w_a = w/w_norm;
    end

    quaternion = [cosd(w_norm/2), sind(w_norm/2)*w_a'];
end

end