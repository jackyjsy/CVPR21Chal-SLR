% script to generate the 2 detection tasks that we address in this paper!
catName = {'bed', 'chair', 'sofa', 'table', 'counter', 'desk', 'lamp', 'pillow', 'sink', 'garbage-bin', 'television','monitor', 'dresser', 'night-stand', 'door', 'bathtub', 'toilet', 'box', 'bookshelf'};
catName = sort(catName);

for i = 1:length(catName),
  task.name{i} = catName{i};
  task.clss{i} = {catName{i}};
end

save('/work5/sgupta/datasets/nyud2/task-guptaetal.mat', '-STRUCT', 'task');
