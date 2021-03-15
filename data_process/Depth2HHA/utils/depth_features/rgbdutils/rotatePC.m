function tmp = rotatePC(pc,R)
% function tmp = rotatePC(pc,R)

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the RGBD Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

  if(isequal(R,eye(3)))
      tmp = pc;
  else
    pc = permute(pc,[3 1 2]);
    tmp = reshape(pc,[3 numel(pc)/3]);
    tmp = R*tmp;
    tmp = reshape(tmp, size(pc));
    tmp = permute(tmp,[2 3 1]);
    pc = tmp;
  end

end
