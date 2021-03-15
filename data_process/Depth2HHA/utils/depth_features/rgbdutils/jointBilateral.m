function out = jointBilateral(refI, I, sigma1, sigma2)
% function out = jointBilateral(refI, I, sigma1, sigma2)

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the RGBD Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
  
  refII = permute(refI, [4 3 2 1]);
  II = permute(I, [4 3 2 1]);
  aa = joint_bilateral_mex(single(II), single(refII), sigma1, sigma2);
  out = double(permute(aa, [4 3 2 1]));
end
