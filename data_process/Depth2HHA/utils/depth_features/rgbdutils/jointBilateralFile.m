function out = jointBilateral(refI, I, sigma1, sigma2, tmpDir, binName)
% function out = jointBilateral(refI, I, sigma1, sigma2, tmpDir, binName)

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the RGBD Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

  if(~exist('tmpDir', 'var')), tmpDir = '/dev/shm/'; end
  if(~exist('binName', 'var')), fileparts(mfilename('fullpath')), 'imagestack', 'bin', 'ImageStack'); end

  assert(isa(refI, 'double'))
  assert(isa(I, 'double'));
  
  % Generate two filenames
  r = randsample(100000, 3, false);
  pid = getPID();
  for i = 1:3,
    f{i} = fullfile(tmpDir, sprintf('rgbdutils-imageStack-%07d-%06d.tmp', pid, r(i)));
  end
    
  refII = isWrite(refI, f{1});
  II = isWrite(I, f{2});
  
  % Run Joint Bilateral Filtering
  % str = sprintf('%s -load %s -load %s -jointbilateral %2.6f %2.6f -save %s double  > /dev/null', binName, f{1}, f{2}, sigma1, sigma2, f{3});
  str = sprintf('time %s -load %s -load %s -time --jointbilateral %2.6f %2.6f -save %s double', binName, f{1}, f{2}, sigma1, sigma2, f{3});
  [a, b] = system(str);
  % b = regexp(b, '\n', 'split'); b = str2num(b{5}(1:5));
  if(a ~= 0)
    % For some reason the bilateral filtering library crashes on some inputs!!
    maxRefI = prctile(linIt(refI(:,:,4)), 98);
    refI(:,:,4) = min(refI(:,:,4), maxRefI);
    out = jointBilateral(refI, I, sigma1, sigma2);
  else
    % Read back the results
    out = isRead(f{3});
  end

  %Remove these files?
  for i = 1:3, str = sprintf('rm %s &', f{i}); system(str); end
  % fprintf(' joint bilateral matlab overhead time: %0.3f\n', toc(tt)-b);
end
