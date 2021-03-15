function dt = getMetadata(fileName)
  c = benchmarkPaths();
  dt = load(sprintf('%s/metadata/%s.mat', c.benchmarkDataDir, fileName));
end
