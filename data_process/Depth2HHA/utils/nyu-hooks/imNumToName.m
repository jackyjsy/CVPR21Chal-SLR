function imname = imNumToName(im, ext)
  imname = arrayfun(@(x) sprintf('img_%04d',x), im, 'UniformOutput', false);
  if(exist('ext', 'var'))
    imname = strcat(imname, '.', ext); 
  end
  if(length(im) == 1)
    imname = imname{1};
  end
end
