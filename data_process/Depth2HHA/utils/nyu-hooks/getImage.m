function I = getImage(imName, typ)
% function I = getImage(imName, typ)
%  Input:
%   typ   one of ['images', 'depth']

% AUTORIGHTS

  paths = benchmarkPaths();
  I = imread(fullfile(paths.dataDir, typ, strcat(imName, '.png')));
end
