function [analysisStats,C] = clonogenicAssay(imgTitle,filePath2,tableName,brightnessThreshold)
%cut out the center region representing the cells in the dish
img1 = imread(imgTitle); %read in image
img = rgb2gray(img1); %convert to grayscale
img_blurred = imgaussfilt(img);
% Get the size of the image
[rows, cols] = size(img_blurred);

% Calculate the radius of the circular region
radius = min(rows, cols) / 2;

% Calculate the center coordinates of the circular region
centerX = round(cols / 2);
centerY = round(rows / 2);

% Create a meshgrid of coordinates
[X, Y] = meshgrid(1:cols, 1:rows);

% Calculate the distance of each point from the center
distances = sqrt((X - centerX).^2 + (Y - centerY).^2);

% Create a logical mask where the values inside the circle are set to true (1)
mask = distances <= radius;

% Create a white background image
background = ones(size(img_blurred), 'like', img_blurred) * 255;

% Apply the mask to the background
img_cropped = img_blurred .* uint8(mask) + background .* uint8(~mask);
 windowSizes = [3, 5, 7, 9];
        
        % Initialize a mask to store the blurry regions
        blurryMask = false(size(img_cropped));
        
        % Calculate variance at different window sizes
        for i = 1:numel(windowSizes)
            windowSize = windowSizes(i);
            
            % Calculate the local variance using imfilter and imgaussfilt
                localVariance = imfilter(double(img_cropped).^2, ones(windowSize)/windowSize^2, 'replicate') - imgaussfilt(double(img_blurred), windowSize).^2;
            
            % Find the blurry regions based on variance values
            blurryRegions = localVariance <= 90;
            
            % Accumulate the blurry regions
            blurryMask = blurryMask | blurryRegions;
        end
        
        % Apply the blurry mask to the original image
        blurryImage = img_cropped;
%         aboveThresholdMask = img_blurred > graythresh(img_blurred);
%          blurryImage(blurryMask & aboveThresholdMask) = 0;  % Set to white

        blurryImage(~blurryMask) = 0; %create an image that displays original
        % grayscale values of blurry regions, and non-blurry regions set to 0 (black)
        blurryBinary = ~blurryImage; %Binary image where white = blurry regions
    
%     brightnessThreshold = 115; % Example threshold value
    
    % Create a binary mask for regions above the brightness threshold
    aboveThresholdMask = img_blurred > brightnessThreshold;
    
    % Set the corresponding regions in 'back' to white
    blurryBinary(aboveThresholdMask) = 1;
    figure, imagesc(blurryBinary), colormap gray
    

    % Apply a threshold to separate foreground and background
    [thresholded, sensitivity] = adaptiveThreshold(img_blurred); 

    thresholded(blurryBinary) = 1;
    bw = thresholded;
    bw2 = ~bw;
    denoised = bwareaopen(bw2, 5);
%     %dilation/erosion to solidify cell arts & get rid of noise
%     se = strel('disk', 5);
%     dilated = imdilate(denoised, se);
%     eroded = imerode(dilated, se);
%     cleaned1 = bwareaopen(eroded, 1000); %get rid of white noise
%     cleaned2 = ~bwareaopen(~cleaned1, 1000); %fill in small black holes
    cleaned2 = denoised;
    % Perform connected component analysis and obtain region properties
    props = regionprops(cleaned2, 'Area');

    % Extract the areas of all objects
    areas = [props.Area];

    % Sort the areas in ascending order
    sortedAreas = sort(areas);

    % Display the sorted sizes of all objects
    %disp(sortedAreas(:));
    areas = sortedAreas(:);

% Determine the threshold area for dividing objects
thresholdArea = 344;

% Count the number of objects
numObjects = numel(props);

% Initialize the rounded count variable
roundedCount = 0;

% Iterate through each object
for i = 1:numObjects
    % Check if the object is larger than the threshold
    if areas(i) > thresholdArea
        % Divide the area by the threshold and round to the nearest integer
        roundedCount = roundedCount + round(areas(i) / 330); %dividng conjoined areas by avg colony size
    else
        % Increment the count for smaller objects
        roundedCount = roundedCount + 1;
    end
end

%naviagte to results folder
C = imfuse(cleaned2, img_blurred, 'montage');

S = extractBefore(imgTitle, '.jpeg');
fileName = "MONTAGE " + S + '.jpeg';
imwrite(C,fileName)

if ~isfile(tableName) %if the table does not already exist
    colNames = {'Photo Name', 'colonies'};
    data = cell(1,2); % create a cell array with one row and the same number of columns as colNames
    table = [cell2table(colNames, 'VariableNames', colNames); cell2table(data, 'VariableNames', colNames)]
    writetable(table, tableName);
    existingTable = readtable(tableName);
    existingTable(1,:) = []; %remove duplicate column names row
else
    existingTable = readtable(tableName); %no need to remove duplicate col name row in this case
end

newRow = {imgTitle, roundedCount}; 


data = [existingTable;newRow]; %add data from analysis
% Find the unique values in PhotoName column
[~, ia, ~] = unique(flipud(data.PhotoName),'rows', 'stable');
ia = size(data,1) - ia + 1;% Select only the rows that contain unique values. If you ran code for
% the same image >1 time, the program will save the most recent version of
analysisStats = flipud(data(ia, :));

writetable(analysisStats, tableName, 'WriteMode', 'overwrite'); %add data to table
% Display the final count
end

%tomorrow after 11