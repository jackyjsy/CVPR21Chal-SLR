function [pc, N, yDir, h, pcRot, NRot] = processDepthImage(z, missingMask, C)
% function [pc, N, yDir, h, pcRot, NRot] = processDepthImage(z, missingMask, C)
% Input: 
%   z is in centimetres
%   C is the camera matrix

% AUTORIGHTS
  addpath('./utils/depth_features/rgbdutils');        % add path into the program.
  yDirParam.angleThresh = [45 15];    % threshold to estimate the direction of the gravity
  yDirParam.iter = [5 5];
  yDirParam.y0 = [0 1 0]';

  normalParam.patchSize = [3 10];

  [X, Y, Z] = getPointCloudFromZ(z, C, 1);
  pc = cat(3, X, Y, Z);   % ç»„æˆ?ä¸‰ä¸ªé€šé?“

  % Compute the normals for this image
  [N1 b1] = computeNormalsSquareSupport(z./100, missingMask, normalParam.patchSize(1),...
    1, C, ones(size(z)));
  [N2 b2] = computeNormalsSquareSupport(z./100, missingMask, normalParam.patchSize(2),...
    1, C, ones(size(z)));
  % [N1 b1] = computeNormals2(pc(:,:,1), pc(:,:,2), pc(:,:,3), ones(size(pc(:,:,1))), normalParam.patchSize(1));
  % [N2 b2] = computeNormals2(pc(:,:,1), pc(:,:,2), pc(:,:,3), ones(size(pc(:,:,1))), normalParam.patchSize(2));
  
  N = N1; 

  % Compute the direction of gravity
  yDir = getYDir(N2, yDirParam);
  y0 = [0 1 0]';
  R = getRMatrix(y0, yDir);

  % rotate the pc and N
  NRot = rotatePC(N, R');
  pcRot = rotatePC(pc, R');
  h = -pcRot(:,:,2);
  yMin = prctile(h(:), 0); 
%   if(yMin > -90) yMin = -90; end
%   yMin = -130
%   yMin
  h = h-yMin;
end
