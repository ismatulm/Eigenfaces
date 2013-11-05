clear, clc, close all

% Eigenfaces and simple classification
% M. Yagci 03.11.13
% images from http://pics.psych.stir.ac.uk/2D_face_sets.htm


%% User input
numImages = 12; % training set size
imTest = imread('data/test5_1.jpg');


%% Read training images and form matrix T where each column is a flattened image
im = imread('data/1.jpg');
r = size(im,1); c = size(im,2);
T = zeros(r*c,numImages);

for i=1:numImages
    im = imread(sprintf('data/%s.jpg',num2str(i)));
    imGray = rgb2gray(im); 
    T(:,i) = reshape(imGray,r*c,1);
end


%% Mean image
m = mean(T');
T = bsxfun(@minus,T,m');
imagesc(reshape(m,r,c)); title('Mean face')
colormap('gray')


%% Costly method (outta memory)
% S = cov(T');
% [V,D] = eig(S);


%% The trick
% recall that actually S=T*T' and we are interested in T*T'*v=lamb*v,
% however this is costly. Instead, try T'*T*u=lamb*u which can be thought
% of as T*T'*T*u=lamb*T*u where v=T*u 
[U,D] = eig(T'*T);
V = T*U;

% visualize eigenfaces
figure
for i=1:numImages
    subplot(2,numImages/2,i)
    imagesc(reshape(V(:,i),r,c))
end
colormap('gray')


%% Simple classification

% Data in eigenspace
% Mind magnitude of eigenvalues when choosing eigenvectors
% I chose all eigenvectors here since the dataset is very small  
data = zeros(numImages,numImages);

% Codebook vectors
for i=1:numImages
    im = imread(sprintf('data/%s.jpg',num2str(i)));
    imGray = rgb2gray(im); 
    imFlat = reshape(imGray,1,r*c);
    imFlat = double(imFlat)-m;
    data(i,:) = imFlat*V;
end

% Test
imGray = rgb2gray(imTest);
imFlat = reshape(imGray,1,r*c);
imFlat = double(imFlat)-m;
testData = imFlat*V;

eucDist = realmax('double');
predictedClass = -1;
for i=1:numImages
    temp = testData-data(i,:);
    dist = (temp*temp')^.5;
    if dist < eucDist
        predictedClass = i;
        eucDist=dist;
    end
end
predictedClass


