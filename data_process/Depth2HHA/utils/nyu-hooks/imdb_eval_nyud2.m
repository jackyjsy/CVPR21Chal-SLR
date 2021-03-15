function res = imdb_eval_nyud2(cls, boxes, imdb)
  [~, clsId] = ismember(cls, imdb.classes);

  % Load the ground truth structures, for the imdb
  parfor i = 1:length(imdb.image_ids),
    % rec = getGroundTruthBoxes(imdb, imdb.image_ids{i});
    rec = getGroundTruthBoxes(imdb, i); 
    
    % filter boxes for the task
    class = {rec.objects(:).class};
    ind = ismember(class, imdb.task.clss{clsId});
    gt(i) = struct('boxInfo', rec.objects(ind), ...
      'diff', [rec.objects(ind).difficult]');
    % tic_toc_print('loading ground truth %05d/%05d.\n', i, length(imdb.image_ids));
  end

  % Do non maximum suppression on the boxes
  parfor i = 1:length(imdb.image_ids);
    bbox = boxes{i};
    keep = rcnn_nms(bbox, 0.3);
    bbox = bbox(keep,:);
    boxInfo = struct('bbox', mat2cell(bbox(:,1:4), ones(size(bbox, 1), 1), 4));
    dt(i) = struct('boxInfo', boxInfo, ...
      'sc', bbox(:, 5));
  end

  % Call the benchmarking function here:
  bOpts.overlapFn = @(x,y,z) bboxOverlap(cat(1, x.boxInfo(:).bbox), cat(1, y.boxInfo(:).bbox));
  bOpts.minoverlap = 0.5;
  bOpts.overlapParam = [];

  [prec, rec, ap, thresh] = instBench(dt, gt, bOpts);
  res.recall = rec;
  res.prec = prec;
  res.ap = ap;
  res.ap_auc = ap;
  res.thresh = thresh;

  % Plot the precision recall curve
  figure(1);
  plot(res.recall, res.prec);
  grid on; ylim([0 1]); xlim([0 1]);
  title(sprintf('%s  AP = %0.3f', cls, res.ap));
  res.plotHandle = gcf;
end
