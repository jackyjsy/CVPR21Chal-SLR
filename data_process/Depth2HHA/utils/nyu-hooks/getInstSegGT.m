function gt = getInstSegGT(imSet, cls, classMapping)
% function gt = getInstSegGT(imSet, cls, classMapping)
%   Returns the inst segmentation masks in format required by instance segment benchmarker
  imList = getImageSet(imSet);
  pt = getMetadata(classMapping);
  pt.className = regexprep(pt.className, ' ', '-');
  clsInd = find(strcmp(pt.className, cls));
  numClass = length(pt.className);
  % cacheDir = fullfile('/work4/sgupta/work2Overflow/detseg/gtCache/', classMapping)

  for i = 1:length(imList),
    % fileName = fullfile(cacheDir, sprintf('%s.mat', imList{i}))
    % try
    %  dt = load(fileName)
    %  gt(i) = dt.gt;
    % catch
      segm = getGroundTruth(imList{i}, 'segmentation', classMapping);
      inst = getGroundTruth(imList{i}, 'instance', classMapping);

      % Find bounding box for each of these instances and their segmentation masks
      freq = cell2mat(accumarray(inst(inst>0), segm(inst>0), [], @(x){linIt(histc(x,1:numClass))'}, {zeros(1,numClass)}))';
      freq = bsxfun(@rdivide, freq, max(1,sum(freq,1)));
      [val, ind] = max(freq,[],1);
      ind(val < 0.5) = 0;
      ind(ind ~= clsInd) = 0;
      instOrig = inst;
      inst(~ismember(inst, find(ind))) = 0;   %% Zero out the not of this class indices
      ind = sort(unique(inst(inst > 0)));
      
      x = repmat([1:size(inst,2)], size(inst,1), 1);
      y = repmat([1:size(inst,1)]', 1, size(inst,2));
      xmin = accumarray(inst(inst>0), x(inst > 0), [], @min);
      xmax = accumarray(inst(inst>0), x(inst > 0), [], @max);
      ymin = accumarray(inst(inst>0), y(inst > 0), [], @min);
      ymax = accumarray(inst(inst>0), y(inst > 0), [], @max);

      ds = [xmin ymin xmax ymax];
      ds = ds(ind, :);
      mask = {};
      instId = [];
      for j = 1:length(ind),
        %% Crop out the portion defined by the box
        mask{j} = inst(ds(j,2):ds(j,4), ds(j,1):ds(j,3)) == ind(j);
        instId(j) = ind(j);
      end
      diff = false(size(ds, 1), 1);

      gt(i) = struct('ds', ds, 'mask', {mask}, 'diff', diff, 'instId', instId, 'instImg', instOrig);
      % parsave(fileName, 'gt', gt(i));
    % end
  end
  
  %% Visualize to make sure all is well
  %  for i = 1:length(imList),
  %    I = getColorImage(imList{i});
  %    figure(1); clf; subplot(1,2,1); hold on; imagesc(I); axis ij
  %    for j = 1:size(gt(i).ds, 1),
  %      subplot(1,2,1); plotDS(gt(i).ds(j,:), 'r');  
  %      subplot(1,2,2); m = zeros(size(I(:,:,1))); m(gt(i).ds(j,2):gt(i).ds(j,4), gt(i).ds(j,1):gt(i).ds(j,3)) = gt(i).mask{j}; imagesc(m);
  %      pause;
  %    end
  %  end
end
