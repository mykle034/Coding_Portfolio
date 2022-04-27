tic

clear all
close all
clc

LeftFrames = dir('Molecule_Video/LeftCamFrames')
LeftFrames = {LeftFrames.name}
RightFrames = dir('Molecule_Video/RightCamFrames')
RightFrames = {RightFrames.name}

LeftFrames(:,1:3) = [];
RightFrames(:,1:2) = [];
error_list = [];

for i = 0:size(RightFrames(:),1)
    
    L = rgb2gray(imread(strcat('Molecule_Video/LeftCamFrames/',LeftFrames{i})));
    R = rgb2gray(imread(strcat('Molecule_Video/RightCamFrames/',RightFrames{i})));


    %Find matching points
    [Lfeatures, Lvalid_points] = extractFeatures(L, detectSURFFeatures(L, 'MetricThreshold', 1000));
    [Rfeatures, Rvalid_points] = extractFeatures(R, detectSURFFeatures(R, 'MetricThreshold', 1000));
    % valid_points.selectStrongest(10)

    indexPairs = matchFeatures(Lfeatures, Rfeatures);
    Lmatched  = Lvalid_points(indexPairs(:,1));
    Rmatched = Rvalid_points(indexPairs(:,2));
%     figure;
%     showMatchedFeatures(L,R,Lmatched,Rmatched);
%     title('Putatively matched points (including outliers)');

    [fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(...
      Lmatched, Rmatched, 'Method', 'RANSAC', ...
      'NumTrials', 2000, 'DistanceThreshold', 0.01, 'Confidence', 99.99);

    if status ~= 0 || isEpipoleInImage(fMatrix, size(L)) ...
      || isEpipoleInImage(fMatrix', size(R))
      warning(['Either not enough matching points were found or '...
             'the epipoles are inside the images. You may need to '...
             'inspect and improve the quality of detected features ',...
             'and/or improve the quality of your images.']);
      error_list = [error_list, i]
      continue
      
    end

    Linlier = Lmatched(epipolarInliers, :);
    Rinlier = Rmatched(epipolarInliers, :);

%     figure;
%     showMatchedFeatures(L, R, Linlier, Rinlier);
%      legend('Inlier points in L', 'Inlier points in R');
% 
    %Rectification
    [t1, t2] = estimateUncalibratedRectification(fMatrix, ...
       Linlier.Location, Rinlier.Location, size(R));
    tform1 = projective2d(t1);
    tform2 = projective2d(t2);
    [Lr, Rr] = rectifyStereoImages(L, R, tform1, tform2);

%     stdisp(Lr, Rr);%Used to determine disparity range

    [di,sim,peak] = istereo(Lr, Rr, [5 150], 3, 'interp');
    hold off;

    temp = di;

    %Ititialize Status
    di = temp;
    status = ones(size(di));
    [U,V] = imeshgrid(di);
    %status(U<=240) = 2;  %Cuts the left black bar, due to camera non-overlap
    status(sim<0.4) = 3; %Only keeps the peaks with similarity >80%
    status(peak.A>=-0.01) = 4; % broad peak, higher keeps more broad peaks (similar textures)
    status(isnan(di)) = 5; %is Nan
    di(status>1) = NaN;

    [U,V] = imeshgrid(di);
    u0 = size(L,2); v0 = size(di,1);
    b = 0.0165;
    f = 0.0046;
    correction = 24;
    rho = 0.0098/4036/correction; %1/2.6 inch: 0.0098m, 4032 pixels in X, 3024 in Y.

    X = b*(U-u0) ./ di; Y = b*(V-v0) ./ di; 
    Z = f/rho * b ./ di; %Z = F/rho

    f = figure;
    surf(Z);
    xlabel("X"); ylabel("Y"), zlabel("Z");
     xlim([0 1850]); ylim([0 1000]);caxis([5 50]); 
    colorbar;
    shading interp; view(0, -90);
    colormap(hot);
    
    exportgraphics(f, strcat('StereoFrames_',num2str((i)*15+1),'.png'));
    
    i/size(RightFrames(:),1)*100
    
    close all
    
    
end

toc
    
