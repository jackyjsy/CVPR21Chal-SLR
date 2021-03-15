function [out, numClass] = getGroundTruth(imName, typ, categoryMap, sceneMap, superpixels)
% function out = getGroundTruth(imName, typ, categoryMap, sceneMap, superpixels)
  c = benchmarkPaths();

  if(isnumeric(imName))
    imName = imNumToName(imName);
  end
  
  switch typ,
    case 'bsdsStruct',
      fileName = fullfile(c.benchmarkGtDir, strcat(imName, '.mat'));
      dt = load(fileName);
      out = dt.groundTruth;
      
    case 'instance',
      fileName = fullfile(c.benchmarkGtDir, strcat(imName, '.mat'));
      dt = load(fileName);
      out = dt.groundTruth{1}.Segmentation;

    case 'segmentation',
      pt = getMetadata(categoryMap);
      numClass = length(pt.className); 
      % fileName = fullfile(c.benchmarkDataDir, 'semanticSegmentation_fixed', strcat(imName, '.mat'));
      % dt = load(fileName, 'gtClass'); 
      
      fileName = fullfile(c.benchmarkGtDir, strcat(imName, '.mat'));
      dt = load(fileName);
      gtClass = dt.groundTruth{1}.SegmentationClass;
      out = double(round(gtClass));
      if(~isequal(out, gtClass))
        fprintf('%s: Ground truth not integral....!\n', imName);
      end
      out(out > 0) = pt.mapClass(out(out > 0));

    case 'sp',
      pt = getMetadata(categoryMap);
      
      fileName = fullfile(c.benchmarkGtDir, strcat(imName, '.mat'));
      dt = load(fileName);
      gtClass = dt.groundTruth{1}.SegmentationClass;
      out = double(round(gtClass));
      if(~isequal(out, gtClass))
        fprintf('%s: Ground truth not integral....!\n', imName);
      end
      out(out > 0) = pt.mapClass(out(out > 0));

      numClass = length(pt.className);
      spHist = cell2mat(accumarray(superpixels(:), out(:),[],@(x){linIt(histc(x,0:numClass))})');
      [val, gt] = max(spHist, [], 1);
      gt = gt-1;
      out = gt;

    case 'scene',
      fileName = fullfile(c.sceneClassFile);
      dt = load(fileName);
      pt = getMetadata(sceneMap);
      out = pt.mapScene(dt.gtScene);
      if(~iscell(imName))
        imName = {imName};
      end
      for i = 1:length(imName),
        im(i) = imNameToNum(imName{i});
      end
      out = out(im-5000);
  end
end
