function [] = main()
part = 2;
if part ==1
    %% Rectify an image
    % csv files contain 4 conresponding sample points to compute H matrix.
    % File name is the image name to be saved
    imname = 'stadium.jpg';
    pointscsv = 'stadiumPoints.csv';
    resultcsv = 'stadiumResults.csv';
    filename = 'stadiumTopView.jpg';
    points = csvread(pointscsv);
    points = [points(:,2), points(:,1)];
    rPoints = csvread(resultcsv);
    rPoints = [rPoints(:,2), rPoints(:,1)];
    
    img = imread(imname);
    H = computeH(points, rPoints);
    result = warpImage(img, H, size(img, 1), size(img, 2));
    
    imwrite(result, filename);
elseif part == 2
    %% Image mosaic
    imname = 'doe.jpg';
    imname2 = 'doe2.jpg';
    im2csv = 'doe2Points.csv';
    basecsv = 'doePoints.csv';
    
    im2pts = csvread(im2csv);
    im2pts = [im2pts(:,2), im2pts(:,1)];
    basepts = csvread(basecsv);
    basepts = [basepts(:,2), basepts(:,1)];
    imbase = imread(imname);
    im2 = imread(imname2);
    H = computeH(im2pts, basepts);
    result = warpImage(im2, H, size(imbase, 1), size(imbase, 2));
    result = result * 255;
%     boo1 = (rgb2gray(result) > 0);
%     boo2 = (rgb2gray(imbase) > 0);
%     mask = boo1 + boo2;
%     intersect = mask > 1;
%     mask(intersect) = 0.5;
    mask = zeros(size(result(:, :, 1)));
    mask(:, size(mask, 2) / 2 : end) = 1;
    final = blend(double(imbase), result, mask);
    imshow(final);
    %imshow(result);
%     imname3 = 'doe3.jpg';
%     
%     im3 = imread(imname3);
%     result = zeros(size(im3, 1) * 3, size(im3, 2) * 3, 3);
%     result(round(size(result, 1) * 0.33):round(size(result, 1) * 0.33) + 599,...
%         round(size(result, 2) * 0.5):round(size(result, 2) * 0.5) + 799, :) = ...
%         im3;
%     imwrite(uint8(result), 'doe.jpg');
end
end

%% Helper functions
function [H] = computeH(im1_pts,im2_pts)
A = zeros(size(im1_pts, 1) * 2, 8);
b = zeros(size(im1_pts, 1) * 2, 1);
for c = 1:size(im1_pts, 1);
    x = im1_pts(c, 1);
    y = im1_pts(c, 2);
    wx = im2_pts(c, 1);
    wy = im2_pts(c, 2);
    A(c * 2 - 1, 1) = x;
    A(c * 2 - 1, 2) = y;
    A(c * 2 - 1, 3) = 1;
    A(c * 2 - 1, 7) = -x * wx;
    A(c * 2 - 1, 8) = -y * wx;
    b(c * 2 - 1) = wx;
    
    A(c * 2, 4) = x;
    A(c * 2, 5) = y;
    A(c * 2, 6) = 1;
    A(c * 2, 7) = -x * wy;
    A(c * 2, 8) = -y * wy;
    b(c * 2) = wy;
end
xResult = ((A' * A) ^ -1) * A' * b;
H = zeros(3, 3);
H(1, 1) = xResult(1);
H(1, 2) = xResult(2);
H(1, 3) = xResult(3);
H(2, 1) = xResult(4);
H(2, 2) = xResult(5);
H(2, 3) = xResult(6);
H(3, 1) = xResult(7);
H(3, 2) = xResult(8);
H(3, 3) = 1;
end

function [imwarped] = warpImage(img, H, sizer, sizec)
    imgr = img(:, :, 1);
    imgg = img(:, :, 2);
    imgb = img(:, :, 3);
    imgr = im2double(imgr);
    imgg = im2double(imgg);
    imgb = im2double(imgb);
    xq = zeros(sizer, sizec);
    yq = zeros(sizer, sizec);
    for r = 1:size(xq, 1)
        prime = [r * ones(1, size(xq, 2)); 1:size(xq, 2); ones(1, size(xq, 2))];
        p = H ^ -1 * prime;
        p(1, :) = p(1, :) ./ p(3, :);
        p(2, :) = p(2, :) ./ p(3, :);
        xq(r, :) = p(2, :);
        yq(r, :) = p(1, :);
    end
    resultr = interp2(imgr, xq, yq);
    resultg = interp2(imgg, xq, yq);
    resultb = interp2(imgb, xq, yq);
    imwarped = cat(3, resultr, resultg, resultb);
end

function [f]= myGaussFilt(img, sigma)
f = conv2(img, gaussian2d(sigma), 'same');
end

function [f] = gaussian2d(sigma)
N = sigma * 2;
[x, y] = meshgrid(round(-N/2):round(N/2), round(-N/2):round(N/2));
f = exp(-x.^2/(2*sigma^2) - y.^2 / (2*sigma^2));
f = f ./ sum(f(:));
end


function [result] = blend(im1, im2, mask1)
im1r = im1(:, :, 1);
im1g = im1(:, :, 2);
im1b = im1(:, :, 3);
im2r = im2(:, :, 1);
im2g = im2(:, :, 2);
im2b = im2(:, :, 3);
mask2 = ones(size(mask1)) - mask1;
N = 1;
lowIm1r = im1r;
lowIm1g = im1g;
lowIm1b = im1b;
lowIm2r = im2r;
lowIm2g = im2g;
lowIm2b = im2b;
sigma = 1;
resultr = zeros(size(im1r));
resultg = zeros(size(im1g));
resultb = zeros(size(im1b));
for a = 1:N
    low1r = myGaussFilt(im1r, sigma);
    low2r = myGaussFilt(im2r, sigma);
    
    low1g = myGaussFilt(im1g, sigma);
    low2g = myGaussFilt(im2g, sigma);
    
    low1b = myGaussFilt(im1b, sigma);
    low2b = myGaussFilt(im2b, sigma);
    
    tempMask1 = myGaussFilt(mask1, sigma);

    tempMask2 = myGaussFilt(mask2, sigma);
    
    sigma = sigma * 2;
    high1r = lowIm1r - low1r;
    high2r = lowIm2r - low2r;
    
    high1g = lowIm1g - low1g;
    high2g = lowIm2g - low2g;
    
    high1b = lowIm1b - low1b;
    high2b = lowIm2b - low2b;
    
    lowIm1r = low1r;
    lowIm2r = low2r;
    
    lowIm1g = low1g;
    lowIm2g = low2g;

    lowIm1b = low1b;
    lowIm2b = low2b;
    
    resultr = resultr + high1r .* tempMask1 + high2r .* tempMask2;
    resultg = resultg + high1g .* tempMask1 + high2g .* tempMask2;
    resultb = resultb + high1b .* tempMask1 + high2b .* tempMask2;
end
resultr = resultr + low1r .* tempMask1 + low2r .* tempMask2;
resultg = resultg + low1g .* tempMask1 + low2g .* tempMask2;
resultb = resultb + low1b .* tempMask1 + low2b .* tempMask2;
result = uint8(cat(3, resultr, resultg, resultb));
end