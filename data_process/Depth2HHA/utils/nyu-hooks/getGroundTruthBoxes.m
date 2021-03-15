function rec = getGroundTruthBoxes(imdb, ii)
  % Load the ground truth segmentation and instances
  try 
    c = benchmarkPaths();
    filename = fullfile_ext(c.gt_box_cache_dir, imdb.image_ids{ii}, 'mat');
    load(filename);
  catch
  im = imdb.image_ids{ii};
  
  % NYU Things here
  inst = getGroundTruth(im, 'instance');
  [segm, numClass] = getGroundTruth(im, 'segmentation','classMappingAll');
  pt = getMetadata('classMappingAll');
  className = pt.className;

  freq = cell2mat(accumarray(inst(inst>0), segm(inst>0), [], @(x){linIt(histc(x,1:numClass))'}, {zeros(1,numClass)}))';
  freq = bsxfun(@rdivide, freq, max(1,sum(freq,1)));
  [val, ind] = max(freq,[],1);
  ind(val < 0.5) = 0;
  
  x = repmat([1:size(inst,2)], size(inst,1), 1);
  y = repmat([1:size(inst,1)]', 1, size(inst,2));
  xmin = accumarray(inst(inst>0), x(inst > 0), [], @min);
  xmax = accumarray(inst(inst>0), x(inst > 0), [], @max);
  ymin = accumarray(inst(inst>0), y(inst > 0), [], @min);
  ymax = accumarray(inst(inst>0), y(inst > 0), [], @max);

  % Make the instances into the rec record 
  class = className(ind(ind > 0));
  difficult = false(1, length(class));
  truncated = false(1, length(class));
  bbox = [xmin'; ymin'; xmax'; ymax'];
  bbox = bbox(:, ind > 0);
  instInd = 1:length(ind); 
  instInd = instInd(ind > 0);

  for j = size(bbox, 2):-1:1,
    rec.objects(j).instanceId = instInd(j);
    rec.objects(j).difficult = difficult(j);
    rec.objects(j).truncated = truncated(j);
    rec.objects(j).bbox = bbox(:,j)';
    rec.objects(j).class = class{j};
  end
  % put these together in the rec as struct array...
  rec.imgsize = [size(inst,1), size(inst,2)];

  save(filename, 'rec');
end
