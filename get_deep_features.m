function [feat,x] = get_deep_features(im, config, cos_window)
%GET_DEEP_FEATURES
%   Extracts dense features from image.

global net
global enableGPU

if isempty(net)
    initial_net();
end

sz_window = size(cos_window);
config.layers = [37,28,19];
% Preprocessing
if size(im,3) == 1
    img(:,:,1)=im(:,:,1);
    img(:,:,2)=im(:,:,1);
    img(:,:,3)=im(:,:,1);
else
    img = im;
end
img = single(img);        % note: [0, 255] range
img = imResample(img, net.normalization.imageSize(1:2));
img = img - net.normalization.averageImage;
if enableGPU, img = gpuArray(img); end

% Run the CNN
res = vl_simplenn(net,img);

% Initialize feature maps
feat = cell(length(config.layers), 1);

for ii = 1:length(config.layers)
    % Resize to sz_window
    if enableGPU
        x = gather(res(config.layers(ii)).x); 
    else
        x = res(config.layers(ii)).x;
    end
    
    x = imResample(x, sz_window(1:2));
    
    % windowing technique
    if ~isempty(cos_window),
        x = bsxfun(@times, x, cos_window);
    end
    
    feat{ii}=x;
end
	
end
