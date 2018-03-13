%% Clear all
clc;
close all;

%% Loading image
listing = dir('data/data/Melanoma/*.jpg');
% listing = {}

for th=5
for i=1:size(listing,1)
    ac_image = imread(strcat('data/data/Melanoma/',listing(i).name));
    image_seg = logical(imread(strcat('data/data/Melanoma/',listing(i).name(1:end-4),'_segmentation.png')));
%     ac_image = imresize(ac_image,[400 400]);
%     image_seg = imresize(image_seg,[400 400]);
    output = remove_scale(ac_image,image_seg,th);
%     imshow(output)
    imwrite(output,strcat('results',char(string(th)),'/Melanoma/',listing(i).name));
    
    %h = figure;set(h, 'Visible', 'off');
    %image(output);
    %axis image
    %axis off
    %set(h, 'LooseInset',get(h,'TightInset'));
    %saveas(h,strcat('results',char(string(th)),'/',listing(i).name(1:end-4),'.eps'),'epsc')

end
end

