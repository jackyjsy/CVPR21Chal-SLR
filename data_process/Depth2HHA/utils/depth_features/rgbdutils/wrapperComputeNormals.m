function [N, b, pc, yDir2] = wrapperComputeNormals(depthImage, missingMask, R, sc, cameraMatrix)
% function [N, b, pc, yDir2] = wrapperComputeNormals(depthImage, missingMask, R, sc, cameraMatrix)
% Expects input to be in metres
% This is how I would call this function
% dt = load('/work4/sgupta/kinect/downloads/nyu_depth_v2v2_labelled.mat', 'depths', 'rawDepths');
% depthImage = double(dt.depths(:,:,1));
% missingMask = dt.rawDepths(:,:,1) == 0;
% [N, b, yDir2] = wrapperComputeNormalsV7(depthImage, missingMask, 3);
% sc is the scale by which the depth image has been upsamples wrt the camera matrix at hand

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the RGBD Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

  if(~exist('sc', 'var'))
    sc = 1;
  end

	%Settings for sigma
	sigmaSupport = R;
	sigmaDisparity = 3;
	sigmaSuperpixel = 10; %Does not really matter
	
	%Convert the depth image into cm
	depthImage = depthImage*100;
	[X, Y, Z] = getPointCloudFromZ(depthImage, cameraMatrix, sc);
  pc = cat(3,X,Y,Z);
	superpixels = ones(size(Z));
	qzc = 2.9e-5;
	disparity = 1./(qzc.*Z);

	% tic;
	[N b] = computeNormals(X, Y, Z, disparity, missingMask, superpixels, sigmaSupport, sigmaDisparity, sigmaSuperpixel);
	% toc;

	% figure(1); subplot(2,2,1); imagesc(visualizeNormals(N));
	% sigmaSupport = 10;
	% [Ngravity b] = computeNormalsV7(X, Y, Z, disparity, missingMask, superpixels, sigmaSupport, sigmaDisparity, sigmaSuperpixel);
	% yDir0 = [0 1 0]';
	% yDir1 = getYDirV3(Ngravity, yDir0, 45, 5);
	% yDir2 = getYDirV3(Ngravity, yDir1, 15, 5);
	% 
	% figure(1); subplot(2,2,2); imagesc((acosd(abs(sum(bsxfun(@times, N, reshape(yDir0, [1 1 3])), 3)))), [0 90]);
	% figure(1); subplot(2,2,3); imagesc((acosd(abs(sum(bsxfun(@times, N, reshape(yDir2, [1 1 3])), 3)))), [0 90]);
end
