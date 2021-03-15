function imdb = imdb_from_nyud2(root_dir, imset, task, regionDir, salt, max_boxes)
  
  % Load the image sets
  imList = getImageSet(imset);
  imdb.image_ids = imList;
  imdb.fold = ones(size(imdb.image_ids));
  % dt = load(fullfile(root_dir, 'splits-rmrc.mat'), imset);
  % imlist = dt.(imset);
  % for i = 1:length(dt.(imset)),
  %   imdb.image_ids{i} = sprintf('img_%04d', imlist(i));
  %   imdb.fold(i) = 1;
  % end
  imdb.imset = imset;

  % Load the class mapping
  imdb.task = getMetadata(task); 
  cls_to_id = containers.Map();
  for i = 1:length(imdb.task.name),
    for j = 1:length(imdb.task.clss{i}),
      cls_to_id(imdb.task.clss{i}{j}) = i;
    end
  end
  imdb.cls_to_id = cls_to_id;
  imdb.classes = imdb.task.name;
  imdb.num_classes = length(imdb.classes);
  imdb.class_ids = 1:imdb.num_classes;

  imdb.name = ['nyud2_' imset '_' salt];
  imdb.dataset_name = ['nyud2_' salt];

  imdb.image_dir = fullfile(root_dir, 'images');
  imdb.depth_dir = fullfile(root_dir, 'depth');
  imdb.rawdepth_dir = fullfile(root_dir, 'rawdepth');

  imdb.image_ext = 'png';
  imdb.depth_ext = 'png';
  imdb.rawdepth_ext = 'png';


  imdb.regionDir = regionDir;
  imdb.roi_func = @(x, y) roi_from_nyud2(x, y);
  imdb.eval_func = @imdb_eval_nyud2;

  imdb.max_boxes = max_boxes;
end
