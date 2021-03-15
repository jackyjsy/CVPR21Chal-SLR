function R = getRMatrix(yi, yf)
% function R = getRMatrix(yi, yf)
% Generates a rotation matrix that
%   if yf is a scalar, rotates about axis yi by yf degrees
%   if yf is an axis, rotates yi to yf in the direction given by yi x yf

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the RGBD Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
  if(isscalar(yf))
    ax = yi./norm(yi);		% norm(A) = max(svd(A))
    phi = yf;
  else
	  yi = yi./norm(yi);
	  yf = yf./norm(yf);
	  ax = cross(yi,yf);
	  ax = ax./norm(ax);
	  %Find angle of rotation
	  % phi = acosd(abs(yi'*yf)); % we dont need to take absolute value here.
	  phi = acosd(yi'*yf);		% get the degree
  end

	if(abs(phi) > 0.1),
		% ax = cross(yi,yf);
		% ax = ax./norm(ax);
		phi = phi*(pi/180);
		S_hat = [ 0 -ax(3) ax(2); ax(3) 0 -ax(1);-ax(2) ax(1) 0];
		R = eye(3) + sin(phi)*S_hat + (1-cos(phi))*(S_hat^2);
	else
		R = eye(3,3);
	end
end
