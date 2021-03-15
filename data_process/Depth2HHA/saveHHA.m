function HHA = saveHHA(imName, C, outDir, D, RD)
% function HHA = saveHHA(imName, C, outDir, D, RD)

% AUTORIGHTS
% C: matrtix
% outDir: save path
% imName: name of picture you want to save as
% D and RD: depth image and corresponding raw-depth image
  addpath('./utils/depth_features');
  D = double(D)/60; %The unit of the element inside D is 'meter'
  missingMask = RD == 0;
%   misingMask = mask;
%   imshow(missingMask)
  [pc, N, yDir, h, pcRot, NRot] = processDepthImage(D*100, missingMask, C);
  angl = acosd(min(1,max(-1,sum(bsxfun(@times, N, reshape(yDir, 1, 1, 3)), 3))));
    
  % Making the minimum depth to be 100, to prevent large values for disparity!!!
  pc(:,:,3) = max(pc(:,:,3), 100); 
  I(:,:,1) = 31000./pc(:,:,3); 
  I(:,:,2) = h;
  I(:,:,3) = (angl+128-90); %Keeping some slack
  I = uint8(I);
  
  % Save if can save
  if(~isempty(outDir) && ~isempty(imName)), imwrite(I, fullfile(outDir, [imName,'.png'])); end
  HHA = I;
end
