function imdb = merge_imdb(imdbs)
  imdb = imdbs{1};
  for i = 1:length(imdbs),
    assert(unique(imdbs{i}.fold) == 1);
    imdb.image_ids = [imdb.images_ids, imdbs{i}.image_ids];
    imdb.fold = [imdb.fold, i*imdbs{i}.fold];
  end
end
