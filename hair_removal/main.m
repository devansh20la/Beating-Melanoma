%% Clear all
clc;
close all;

%% Loading image
listing = dir('data/data/Keratosis/*.jpg');
% listing = {}

for th=0.1;
for i=1:size(listing,1)
    ac_image = imread(strcat('data/data/Keratosis/',listing(i).name));
    image_seg = logical(imread(strcat('data/data/Keratosis/',listing(i).name(1:end-4),'_segmentation.png')));
    ac_image = imresize(ac_image,[400 400]);
    image_seg = imresize(image_seg,[400 400]);
    output = my_function(ac_image,image_seg,th);
%     imshow(output)
    imwrite(output,strcat('results','/Keratosis/',listing(i).name));
    
    %h = figure;set(h, 'Visible', 'off');
    %image(output);
    %axis image
    %axis off
    %set(h, 'LooseInset',get(h,'TightInset'));
    %saveas(h,strcat('results/',listing(i).name(1:end-4),'.eps'),'epsc')

end
end

