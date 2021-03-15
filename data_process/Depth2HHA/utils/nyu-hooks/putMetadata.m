function getMetadata(fileName, dt)
  c = benchmarkPaths(0);
  fileName = sprintf('%s/metadata/%s.mat', c.benchmarkDataDir, fileName);
  if(~exist(fileName, 'file'))
    save(fileName, '-STRUCT', 'dt');
  else
    error(sprintf('%s already exists!', fileName));
  end
end
