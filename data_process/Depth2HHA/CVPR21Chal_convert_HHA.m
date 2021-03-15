% clc;
close all;
clear;

addpath('./utils/nyu-hooks');
input_folder = '../../data/train/';
hha_root = '../../data/train_hha/';

C = getCameraParam('color');
matrix = C;  

folder = [input_folder, '*depth.mp4'];
filelist = dir(folder);

for i = 1:length(filelist)
    video_name = filelist(i).name;
    video_path = [input_folder, video_name];
    
    save_folder = [hha_root, video_name(1:end-10)];
    mkdir(save_folder);
    
    v = VideoReader(video_path);
    frames = read(v, [1, Inf]);
    [~,~,~,frame_num] = size(frames);
    
    for frame_loop = 1:frame_num
        each_frame = frames(:, :, :, frame_loop);
        frame_gray = rgb2gray(each_frame);
        disp(['video = ', num2str(i), '  frame = ', num2str(frame_loop)]);
        hha = saveHHA(['frame', mat2str(frame_loop)], matrix, save_folder, frame_gray, frame_gray); 
    end
end












