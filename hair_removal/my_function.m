function output = remove_scale(image,image_seg,th)
    
    image_actual = image;
%     r1 = makecform('srgb2xyz');
%     r2 = makecform('xyz2uvl');
% 
%     r1_rev = makecform('uvl2xyz');
%     r2_rev = makecform('xyz2srgb');

    image = rgb2lab(image);
%     image = applycform(image,r1);
%     image = applycform(image,r2);
    l_before = image(:,:,1);

    %% Perform closing
    se = strel('sphere',5);
    image_close = imclose(image,se);
    l_after = image_close(:,:,1);
    new_l = l_after - l_before;
    new_l = wthresh(new_l,'h',th);
    new_l = imbinarize(new_l);
    
%     %new_l(new_l > th) = 1;
%     new_l(new_l <= th) = 0;

    final_1 = new_l.*l_after;
    final_2 = l_before.*~new_l;
    final = final_1 + final_2;

    image(:,:,1) = final;
    image = lab2rgb(image);
%     image = applycform(image,r1_rev);
%     image = applycform(image,r2_rev);

    %% Using segmentation mask
    image = image.*~image_seg;
    temp = image_actual.*uint8(image_seg);

    final = image + double(temp)/255;
    output = final;
%     output = zeros(400,805,3);
%     output(:,1:400,:) = double(image_actual)/255;
%     output(:,406:805,:) = final;
