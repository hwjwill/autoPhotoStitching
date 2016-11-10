function [] = main()
part = 1;
if part ==1
    %% Rectify an image
    imname = 'stadium.jpg';
    pointscsv = 'stadiumPoints.csv';
    resultCSV = 'stadiumResults.csv';
    points = csvread(pointscsv);
    points = [points(:,2), points(:,1)];
    rPoints = csvread(resultCSV);
    rPoints = [rPoints(:,2), rPoints(:,1)];
    
    img = imread(imname);
    img = im2single(rgb2gray(img));
    H = computeH(points, rPoints);
    result = zeros(size(img));
    
    for r = 1:size(img, 1)
        prime = [r * ones(1, size(img, 2)); 1:size(img, 2); ones(1, size(img, 2))];
        p = H ^ -1 * prime;
        result(r, :) = interp2(img, p(2, :), p(1, :));
    end
    imshow(result);
end
end

%% Helper functions
function [H] = computeH(im1_pts,im2_pts)
A = zeros(size(im1_pts, 1) * 3, 7);
b = zeros(size(im1_pts, 1) * 3, 1);
for c = 1:size(im1_pts, 1);
    x = im1_pts(c, 1);
    y = im1_pts(c, 2);
    wx = im2_pts(c, 1);
    wy = im2_pts(c, 2);
    A(c * 3 - 2, 1) = x;
    A(c * 3 - 2, 2) = y;
    A(c * 3 - 2, 3) = 1;
    b(c * 3 - 2) = wx;
    
    A(c * 3 - 1, 4) = x;
    A(c * 3 - 1, 5) = y;
    A(c * 3 - 1, 6) = 1;
    b(c * 3 - 1) = wy;
    
    A(c * 3, 7) = x;
    A(c * 3, 8) = y;
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