function y = getYDir(N, yDirParam)
% function y = getYDir(N, yDirParam)
% Input:
%   N:            normal field
%   yDirParam:    parameters
%                 struct('y0', [0 1 0]', 'angleThresh', [45 15], 'iter', [5 5]);
% Output:
%   y:            Gravity direction

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the RGBD Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

	y = yDirParam.y0;
	for i = 1:length(yDirParam.angleThresh),
		y = getYDirHelper(N, y, yDirParam.angleThresh(i), yDirParam.iter(i));
	end
end

function yDir = getYDirHelper(N, y0, thresh, iter)
% function yDir = getYDirHelper(N, y0, thresh, iter)
%Input: 
%	N: HxWx3 matrix with normal at each pixel.
%	y0: the initial gravity direction
%	thresh: in degrees the threshold for mapping to parallel to gravity and perpendicular to gravity
% 	iter: number of iterations to perform
%Output:
%	yDir: the direction of gravity vector as inferred

	nn = permute(N,[3 1 2]);     		% change the third dimension to the first-order. (480, 680, 3) => (3, 480, 680)
	% 3 * x, which 'x' is the number of the point cloud set.
	nn = reshape(nn,[3 numel(nn)/3]);	% numel: return number of elements in  a matrix.
	nn = nn(:,~isnan(nn(1,:))); 		% remove these whose number is NAN 
	
	%Set it up as a optimization problem.

	yDir = y0;
	%Let us do hard assignments
	for i = 1:iter,
		sim0 = yDir'*nn;
		indF = abs(sim0) > cosd(thresh);		% calculate 'floor' set.    |sin(theta)| < sin(thresh) ==> |cos(theta)| > cos(thresh)
 		indW = abs(sim0) < sind(thresh);		% calculate 'wall' set.

		NF = nn(:, indF);
		NW = nn(:, indW);
		A = NW*NW' - NF*NF';
		b = zeros(3,1);
		c = size(NF,2);

		[V D] = eig(A);		% V: eigenvectors		D: eigenvalues in its diagonal line
		[gr ind] = min(diag(D));
		newYDir = V(:,ind);
		yDir = newYDir.*sign(yDir'*newYDir);		% sign(negative) = -1; sign(positive) = 1;
	end
end
