clear
clc
close all
addpath('helper_functions')

%% Setup
% path to the images folder
path_img_dir = '../data/detection';
% path to object ply file
object_path = '../data/teabox.ply';

% Read the object's geometry 
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);

% Load the SIFT model from the previous task
load('sift_model.mat');


% TODO: setup camera intrinsic parameters using cameraParameters()
% A - The camera instrinsic matrix
fx = 2960.37845;
fy = 2960.37845;
cx = 1841.68855;
cy = 1235.23369;
A = [fx, 0, 0; 0, fy, 0; cx, cy, 1];

image_size = [2456, 3680];
camera_params = cameraParameters("IntrinsicMatrix",A, "ImageSize",image_size);

%% Get all filenames in images folder

FolderInfo = dir(fullfile(path_img_dir, '*.JPG'));
Filenames = fullfile(path_img_dir, {FolderInfo.name} );
num_files = length(Filenames);


%% Match SIFT features of new images to the SIFT model with features computed in the task 1
% You should use VLFeat function vl_ubcmatch()

% Place SIFT keypoints and descriptors of new images here
keypoints=cell(num_files,1);
descriptors=cell(num_files,1);
% Place matches between new SIFT features and SIFT features from the SIFT
% model here
sift_matches=cell(num_files,1);

% Default threshold for SIFT keypoints matching: 1.5 
% When taking higher value, match is only recognized if similarity is very high
threshold_ubcmatch = 2.0; % default 1.5 - increase to filter points, atlternative - SCORES

% for i=1:num_files
%     fprintf('Calculating and matching sift features for image: %d \n', i)
%     
% %     TODO: Prepare the image (img) for vl_sift() function
% %   Covert images to grayscale as required by vl_sift()
%     gray_img = rgb2gray(imread(Filenames{i}));
% %   Convert to single precision matrix
%     img = single(gray_img);
%     [keypoints{i}, descriptors{i}] = vl_sift(img);
% %     Match features between SIFT model and SIFT features from new image
%     sift_matches{i} = vl_ubcmatch(descriptors{i}, model.descriptors, threshold_ubcmatch); 
% end


% Save sift features, descriptors and matches and load them when you rerun the code to save time
% save('sift_matches.mat', 'sift_matches');
% save('detection_keypoints.mat', 'keypoints')
% save('detection_descriptors.mat', 'descriptors')

load('sift_matches.mat')
load('detection_keypoints.mat')
load('detection_descriptors.mat')

%% PnP and RANSAC 
% Implement the RANSAC algorithm featuring also the following arguments:
% Reprojection error threshold for inlier selection - 'threshold_ransac'  
% Number of RANSAC iterations - 'ransac_iterations'

% Pseudocode
% i Randomly select a sample of 4 data points from S and estimate the pose using PnP.
% ii Determine the set of data points Si from all 2D-3D correspondences 
%   where the reprojection error (Euclidean distance) is below the threshold (threshold_ransac). 
%   The set Si is the consensus set of the sample and defines the inliers of S.
% iii If the number of inliers is greater than we have seen so far,
%   re-estimate the pose using Si and store it with the corresponding number of inliers.
% iv Repeat the above mentioned procedure for N iterations (ransac_iterations).

% For PnP you can use estimateWorldCameraPose() function
% but only use it with 4 points and set the 'MaxReprojectionError' to the
% value of 10000 so that all these 4 points are considered to be inliers

% Place camera orientations, locations and best inliers set for every image here
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);
best_inliers_set = cell(num_files, 1);

ransac_iterations = 100; 
threshold_ransac = 4;

random_points_count = 4;
max_reproj_err = 1000;

for i = 1:num_files
    fprintf('Running PnP+RANSAC for image: %d \n', i)
    % Get all the keypoint matches between Image and Model 
    % Give me all the 2D (x,y) of center of frame for the matched keypoints
    sift_matches_image = keypoints{i}(1:2, sift_matches{i}(1, :));
    % Give me all the 3D (x,y,z) of the corresponding matched keypoints
    sift_matches_model = model.coord3d(sift_matches{i}(2, :), :);
        
    % Prepare sift_matches_model for projection onto the image --
    % Homogeneous Cordinates
    sift_matches_model(:, 4) = 1;
   
%     TODO: Implement the RANSAC algorithm here
    best_inliers_set_size = 3;
    for itr = 1:ransac_iterations
        % give me 4 random interger valued indexes from the total sift matches
        sampled_points = randperm(size(sift_matches{i}, 2), random_points_count);
        % give me the actual matched keypoints; matched_keypoints(1,:) is the
        % keypoint number in Image which matched matched_keypoints(2,:) in the
        % model
        matched_keypoints = sift_matches{i}(:, sampled_points);
        index_keypoints_image = matched_keypoints(1, :);
        index_keypoints_model = matched_keypoints(2, :);

        % Get the center of the frame of the keypoints of Image
        image_x_y = keypoints{i}(1:2, index_keypoints_image);
        
        % Similarly get the corresponding world locations from the Sift Model
        world_x_y = model.coord3d(index_keypoints_model, :);
        % Estimate the pose using PnP
        try
            [cam_in_world_orientations_local, cam_in_world_locations_local] = estimateWorldCameraPose(image_x_y', world_x_y, camera_params, 'MaxReprojectionError', max_reproj_err);
            % Sometimes this configurations results in no inliers
        catch ME
            continue
        end
        
        % Convert camera pose to extrinsics
        [rotationMatrix, translationVector] = cameraPoseToExtrinsics(cam_in_world_orientations_local, cam_in_world_locations_local);
        
        % Compute camera projection matrix
        camMatrix = cameraMatrix(camera_params, rotationMatrix, translationVector);
        
        % Project onto image
        camera_cordinates = sift_matches_model * camMatrix;
        
        % get rid of z, project to z=0
        camera_cordinates_z = camera_cordinates(:, 3);
        image_cordinates = camera_cordinates(:, 1:2) ./ camera_cordinates_z;
        
        % Get the reprojection error
        reprojection_error = sift_matches_image - image_cordinates'; % alternative pdist2
        direction_vector = vecnorm(reprojection_error, 2, 1);
        
        % Get inliers indexes
        inliers_index = find(direction_vector < threshold_ransac);
        
        % Update inliers
        if (numel(inliers_index) > best_inliers_set_size)
            best_inliers_set{i} = inliers_index;
            best_inliers_set_size = numel(best_inliers_set{i});
        end
    end
    
    % Restimate the pose to store best pose with best inlier set
    [cam_in_world_orientations(:,:,i),cam_in_world_locations(:,:,i)] = estimateWorldCameraPose(sift_matches_image(:, best_inliers_set{i})', sift_matches_model(best_inliers_set{i}, 1:3), camera_params, ...
        'MaxReprojectionError', max_reproj_err); % images were from 10000
    
end
%% Visualize inliers and the bounding box

% You can use the visualizations below or create your own one
% But be sure to present the bounding boxes drawn on the image to verify
% the camera pose

edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];

for i=1:num_files
    
    figure()
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
    title(sprintf('Image: %d', i))
    hold on
    
%   Plot inliers set
    PlotInlierOutlier(best_inliers_set{i}, camera_params, sift_matches{i}, model.coord3d, keypoints{i}, cam_in_world_orientations(:,:,i), cam_in_world_locations(:,:,i))
%   Plot bounding box
    points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
    for j=1:12
        plot(points(1, edges(:, j)), points(2, edges(:,j)), 'color', 'b');
    end
    hold off;
end