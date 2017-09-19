function [ pos, max_response ] = do_correlation_deep( im, pos, window_sz, cos_window, config, model)

% if size(im,3) > 1, im = rgb2gray(im); end

% cell_size = config.features.cell_size;
% 
% patch = get_subwindow(im, pos, window_sz);          
%             
% zf = fft2(get_deep_features(patch,config,cos_window));		
% 
% kzf = gaussian_correlation(zf, model.xf, config.kernel_sigma);
% 
% response = fftshift(real(ifft2(model.alphaf .* kzf)));
% 
% max_response = max(response(:));
% 
% [vert_delta, horiz_delta] = find(response == max_response, 1); 
% 
% pos = pos + cell_size * [vert_delta - floor(size(zf,1)/2)-1, horiz_delta - floor(size(zf,2)/2)-1];       
indLayers = config.indLayers;
res_layer = zeros([config.l1_patch_num, length(config.indLayers)]);
res_layer_weights = zeros([config.l1_patch_num, length(config.indLayers)]);
patch = get_subwindow(im,pos,window_sz);

[feat,~] = get_deep_features(patch,config,cos_window);

for ii = 1 : length(indLayers)
    zf = fft2(feat{ii});
    kzf=sum(zf .* conj(model.xf{ii}), 3) / numel(zf);
    
    res_layer(:,:,ii) = real(fftshift(ifft2(model.alphaf{ii} .* kzf)));  %equation for fast detection
    res_layer_weights(:,:,ii) = res_layer(:,:,ii)*config.nweights(ii);
end

% Combine responses from multiple layers (see Eqn. 5)
% response = sum(bsxfun(@times, res_layer, config.nweights), 3);
response = sum(res_layer_weights,3);

% ================================================================================
% Find target location
% ================================================================================
% Target location is at the maximum response. we must take into
% account the fact that, if the target doesn't move, the peak
% will appear at the top-left corner, not at the center (this is
% discussed in the KCF paper). The responses wrap around cyclically.
max_response = max(response(:));
[vert_delta, horiz_delta] = find(response == max_response, 1);
vert_delta  = vert_delta  - floor(size(zf,1)/2);
horiz_delta = horiz_delta - floor(size(zf,2)/2);

% Map the position to the image space
pos = pos + config.features.cell_size * [vert_delta - 1, horiz_delta - 1];

end