function [] = main()
part = 3;
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
    % imname is the image name of the base image (the image that is not 
    % going to be wraped). Imname2 is the image name of wrap image. CSV
    % files contain the conrespondence. 
    imname = 'ms.jpg';
    imname2 = 'm3.jpg';
    im2csv = 'm3Points.csv';
    basecsv = 'msPoints.csv';
    filename = 'mStitch.jpg';
    
    im2pts = csvread(im2csv);
    im2pts = [im2pts(:,2), im2pts(:,1)];
    basepts = csvread(basecsv);
    basepts = [basepts(:,2), basepts(:,1)];
    imbase = imread(imname);
    im2 = imread(imname2);
    H = computeH(im2pts, basepts);
    result = single(warpImage(im2, H, size(imbase, 1), size(imbase, 2)));
    result(isnan(result)) = 0;
    mask = zeros(size(result(:, :, 1)));
    mask(:, 1 : size(mask, 2) / 2) = 1;
    final = blend(im2single(imbase), result, mask);
    imwrite(final, filename);
elseif part == 3
    %% Autostitching
    imname1 = 'm1.jpg';
    imname2 = 'car.jpg';
    filename = 'autoMM.jpg';
    im1 = imread(imname1);
    im2 = imread(imname2);
    
%     black = zeros(size(im2, 1) * 3, size(im2, 2) * 3, 3);
%     black(size(black, 1) / 3 : size(black, 1) / 3 + size(im2, 1) - 1,...
%         size(black, 2) / 3 : size(black , 2) / 3 + size(im2, 2) - 1, :) = im2;
%     im2 = uint8(black);
    
    [x1, y1, v1] = harris(im1);
    [x2, y2, v2] = harris(im2);
    
%     figure(1);
%     imshow(im1);
%     hold on;
%     plot(x1,y1,'r.', 'markersize', 15);
%     hold off;
%     figure(2);
%     imshow(im2);
%     hold on;
%     plot(x2,y2,'r.', 'markersize', 15);
%     hold off;  

    % Step 1: Implement Adaptive Non-Maximal Supression
    [x1, y1] = anms(x1, y1, v1, 250);
    [x2, y2] = anms(x2, y2, v2, 250);

%     figure(1);
%     imshow(im1);
%     hold on;
%     plot(x1,y1,'r.', 'markersize', 15);
%     hold off;
%     figure(2);
%     imshow(im2);
%     hold on;
%     plot(x2,y2,'r.', 'markersize', 15);
%     hold off;

    % Step 2: Feature descriptor extraction
    im1gray = rgb2gray(im1);
    im2gray = rgb2gray(im2);
    [d1, xd1, yd1] = extract(x1, y1, im1gray);
    [d2, xd2, yd2] = extract(x2, y2, im2gray);
    
    % Step 3: Feature Matching
    [pts1, pts2] = match(d1, xd1, yd1, d2, xd2, yd2);
    
%     figure(1);
%     imshow(im1);
%     hold on;
%     plot(x1,y1,'r.', 'markersize', 15);
%     plot(pts1(:, 1), pts1(:, 2),'b.', 'markersize', 20);
%     hold off;
%     figure(2);
%     imshow(im2);
%     hold on;
%     plot(x2,y2,'r.', 'markersize', 15);
%     plot(pts2(:, 1), pts2(:, 2),'b.', 'markersize', 20);
%     hold off;    

    % Step 4: RANSAC
    [H] = ransac(pts1, pts2);
    
    % Step 5: Blend
    result = single(warpImage(im1, H, size(im2, 1), size(im2, 2)));
    result(isnan(result)) = 0;
    mask = zeros(size(result(:, :, 1)));
    mask(:, 1 : size(mask, 2) * 2 / 3 - 40) = 1;
    final = blend(im2single(im2), result, mask);
    imshow(final);
    imwrite(final, filename);
elseif part == 4
    %% Panorama recognition
    imnames = ['m1.jpg', 'car.jpg', 'doe3.jpg', 'm2.jpg', 'doe4.jpg'];
    combos = nchoosek(size(imnames, 2), 2);
    for a = 1:combos
        
    end
end
end

%% Helper functions
function [H] = ransac(pts1, pts2)
iterations = 100;
pts1 = [pts1(:, 2), pts1(:, 1)];
pts2 = [pts2(:, 2), pts2(:, 1)];
table = zeros(iterations, size(pts1, 1) + 1);
for a = 1:iterations
   sample = randperm(size(pts1, 1), 4);
   table(a, 2 : 5) = sample;
   train1 = [pts1(sample(1), :); pts1(sample(2), :);...
       pts1(sample(3), :); pts1(sample(4), :)];
   train2 = [pts2(sample(1), :); pts2(sample(2), :);...
       pts2(sample(3), :); pts2(sample(4), :)];   
   
   candidateH = computeH(train1, train2);
   count = 4;
   for b = 1:size(pts1, 1)
       if any(sample == b)
           continue;
       end
       rp = transform(pts1(b, :), candidateH);
       if comp(rp, pts2(b, :), 1)
           count = count + 1;
           table(a, count + 1) = b;
       end
   end
   table(a, 1) = count;
end
[c, ind] = max(table(:, 1));
from = zeros(c, 2);
to = zeros(c, 2);
for j = 2 : c + 1
    curr = table(ind, j);
    from(j - 1, :) = pts1(curr, :);
    to(j - 1, :) = pts2(curr, :);
end
H = computeH(from, to);
end

function [rp] = transform(pt, H)
x = [pt(1); pt(2); 1];
y = H * x;
y = y / y(3);
rp = [y(1); y(2)];
end

function [close] = comp(pt1, pt2, thres) 
d = sqrt((pt1(1) - pt2(1)) ^ 2 + (pt1(2) - pt2(2)) ^ 2);
if d < thres
    close = true;
else
    close = false;
end
end

function [pts1, pts2] = match(d1, xd1, yd1, d2, xd2, yd2)
ssdTable = zeros(size(d1, 1), size(d2, 1));
thresh = 0.3;
for a = 1 : size(ssdTable, 1)
    for b = 1 : size(ssdTable, 2)
        ssdTable(a, b) = ssd(d1(a, :), d2(b, :));
    end
end
nn1 = zeros(size(d1, 1), 1);
nn2 = zeros(size(d1, 1), 1);
potentialMatches = zeros(size(d1, 1), 1);
for c = 1:size(nn1, 1)
    [nn1(c), potentialMatches(c)] = min(ssdTable(c, :));
    ssdTable(c, potentialMatches(c)) = 99999;
    nn2(c) = min(ssdTable(c, :));
end
ratio = nn1 ./ nn2;
selector = ratio < thresh;
pts1 = [xd1(selector), yd1(selector)];
potentialMatches = potentialMatches(selector);
pts2 = size(pts1);
for d = 1:size(potentialMatches, 1)
    pts2(d, :) = [xd2(potentialMatches(d)), yd2(potentialMatches(d))];
end
end

function [val] = ssd(array1, array2)
val = sum((array1 - array2) .^ 2);
end

function [descriptor, xd, yd] = extract(c, r, im)
scale = 5;
descriptor = zeros(size(r, 1), 64);
zerorows = zeros(size(r, 1), 1);
for a = 1:size(r, 1)
    rmin = r(a) - 4 * scale;
    rmax = r(a) + 4 * scale;
    cmin = c(a) - 4 * scale;
    cmax = c(a) + 4 * scale;
    if rmin < 1 || cmin < 1 || rmax > size(im, 1) || cmax > size(im, 2)
        zerorows(a) = 1;
        continue; 
    end
    patch = double(im(rmin : rmax -  1, cmin : cmax - 1));
    patch = myGaussFilt(patch, scale);
    temp = zeros(8, 8);
    for count1 = 1:8
        for count2 = 1:8
            temp(count1, count2) = patch(count1 * scale, count2 * scale);
        end
    end
    patch = reshape(temp, [1, size(temp, 1) * size(temp, 2)]);
    avg = mean(patch);
    sd = std(patch);
    patch = patch - avg;
    patch = patch / sd;
    descriptor(a, :) = patch;
end
zerorows = zerorows ~= 0;
descriptor(zerorows, :) = [];
xd = c;
yd = r;
xd(zerorows) = [];
yd(zerorows) = [];
end

function [xmax, ymax] = anms(x, y, v, count)
maxRadius = zeros(size(x));
[~, maxi] = max(v);
maxRadius(maxi) = 99999;
for a = 1:size(x, 1)
   if a == maxi
       continue;
   end
   r = 1;
   val = maxAlongLine(x, y, v, r, a);
   while val < v(a)
       r = r + 1;
       val = maxAlongLine(x, y, v, r, a);
   end
   maxRadius(a) = r;
end
sorted = sort(maxRadius, 'descend');
sorted = sorted(1:count);
indexes = zeros(count, 1);
b = 1;
while b <= count
    found = find(maxRadius == sorted(b));
    l = size(found, 1) - 1;
    e = b + l;
    if e >= count
       e = count;
       indexes(b : e) = found(1: e - b + 1);
    else
       indexes(b : e) = found;
    end
    b = b + l + 1;
end
xmax = x(indexes);
ymax = y(indexes);
end

function [value] = maxAlongLine(x, y, v, r, i)
currX = x(i);
currY = y(i);
xmin = currX - r;
xmax = currX + r;
ymin = currY - r;
ymax = currY + r;
log1 = x == xmin & y <= ymax & y >= ymin;
log2 = x == xmax & y <= ymax & y >= ymin;
log3 = y == ymin & x <= xmax & x >= xmin;
log4 = y == ymax & x <= xmax & x >= xmin;
logic = log1 | log2 | log3 | log4;
if any(logic) == 0
    value = 0;
else
    value = max(v(logic));
end
end

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
N = 2;
lowIm1r = im1r;
lowIm1g = im1g;
lowIm1b = im1b;
lowIm2r = im2r;
lowIm2g = im2g;
lowIm2b = im2b;
sigma = 10;
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
result = cat(3, resultr, resultg, resultb);
end
