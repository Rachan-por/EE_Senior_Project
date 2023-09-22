clear all; close all; clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code is for resizing images using Bicubic interpolation.          %
% The original of this code is from https://github.com/yulunzhang/RCAN.  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1. set path 
path_original = 'C:\Users\USER\Desktop'; % Path that has dataset folder 
dataset  = {'Set5'}; % dataset
ext = {'.png'}; % file extension

% 2. set scale that will resize
scale_all = [2,3,4,8];
degradation = 'BI'; % BI, BD, DN

% 3. rescale
for idx_set = 1:length(dataset) % iterate every dataset
    fprintf('Processing %s:\n', dataset{idx_set});
    filepaths = []; % store file paths
    for idx_ext = 1:length(ext) % iterate every file extensions
        directory = dir(fullfile(path_original, dataset{idx_set})); % create filename of images
        filepaths = cat(1, filepaths, directory); % concatenate image file information
    end
    for idx_im = 4:length(filepaths) % iterate every images
        name_im = filepaths(idx_im).name; % get name of the file
        fprintf('%d. %s: ', idx_im, name_im); 
        file_path = fullfile(path_original, dataset{idx_set}, name_im); % image file
        im_ori = imread(file_path); % read images
        if size(im_ori, 3) == 1 % read images
            im_ori = cat(3, im_ori, im_ori, im_ori); % for 1 layer image, change to 3 layer for easier manipulation
        end
        for scale = scale_all % iterate every scale
            fprintf('x%d ', scale);
            im_HR = modcrop(im_ori, scale); % modify image until it is divisible
            if strcmp(degradation, 'BI') % 
                im_LR = imresize(im_HR, 1/scale, 'bicubic');
            elseif strcmp(degradation, 'BD')
                im_LR = imresize_BD(im_HR, scale, 'Gaussian', 1.6); % sigma=1.6
            elseif strcmp(degradation, 'DN')
                randn('seed',0); % For test data, fix seed. But, DON'T fix seed, when preparing training data.
                im_LR = imresize_DN(im_HR, scale, 30); % noise level sigma=30
            end
         
            folder_HR = fullfile('./HR', dataset{idx_set}, ['x', num2str(scale)]); % create HR folder
            folder_LR = fullfile(['./LR/LR', degradation], dataset{idx_set}, ['x', num2str(scale)]); % create LR folder
            if ~exist(folder_HR) % check if it exists
                mkdir(folder_HR)
            end
            if ~exist(folder_LR) % check if it exists
                mkdir(folder_LR)
            end
            fn_HR = fullfile('./HR', dataset{idx_set}, ['x', num2str(scale)], [name_im(1:end-4), '.png']); % name of HR image file
            fn_LR = fullfile(['./LR/LR', degradation], dataset{idx_set}, ['x', num2str(scale)], [name_im(1:end-4), '.png']); % name of LR image file
            imwrite(im_HR, fn_HR, 'png'); % save image file
            imwrite(im_LR, fn_LR, 'png'); % save image file
        end
        fprintf('\n');
    end
    fprintf('\n');
end

% Helper function to modify image size when it is not divisible
function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1 % if image has 1 layer
    sz = size(imgs);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2));
else % if image has 3 layer
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2),:);
end
end