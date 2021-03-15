function c = benchmarkPaths()
  c.dataRootDir = fullfile(pwd(), '../eccv14-data');
  assert(exist(c.dataRootDir, 'dir') > 0, sprintf('%s: data directory does not exist!\n', c.dataRootDir));

    c.benchmarkDataDir = fullfile(c.dataRootDir, 'benchmarkData');
    assert(exist(c.benchmarkDataDir, 'dir') > 0, sprintf('%s: benchmark data directory does not exist!\n', c.benchmarkDataDir));

      c.benchmarkGtDir = fullfile(c.benchmarkDataDir, 'groundTruth');
      assert(exist(c.benchmarkGtDir, 'dir') > 0, sprintf('%s: benchmark data directory does not exist!\n', c.benchmarkGtDir));

      c.gt_box_cache_dir = fullfile(c.benchmarkDataDir, 'gt_box_cache_dir');
      exists_or_mkdir(c.gt_box_cache_dir);

    c.dataDir = fullfile(c.dataRootDir, 'data');
    assert(exist(c.dataDir, 'dir') > 0, sprintf('%s: benchmark data directory does not exist!\n', c.dataDir));

end

function made = exists_or_mkdir(path)
% function made = exists_or_mkdir(path)
% Make directory path if it does not already exist.

% Obtained from voc-release5

  made = false;
  if exist(path) == 0
    unix(['mkdir -p ' path]);
    made = true;
  end
end
