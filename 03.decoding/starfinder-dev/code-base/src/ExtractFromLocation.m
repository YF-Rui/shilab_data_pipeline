function [color_seq, color_score] = ExtractFromLocation( input_img, allSpots, voxel_size )
%ExtractFromLocation

    % get dims
    [dimX, dimY, dimZ, Nchannel] = size(input_img);
    Npoint = size(allSpots, 1);
    color_matrix = zeros(Npoint, Nchannel); % "color value" of each dot in each channel of each sequencing round 
    color_seq = string(zeros(Npoint, 1));
    color_score = zeros(Npoint, 1);

    % disp("===Performing imdilate===")
    % se = strel('sphere', 1);
    % for c = 1:4
    %     input_img(:,:,:,c) = gather(imdilate(gpuArray(input_img(:,:,:,c)), se));
    % end
    reshaped = reshape(double(input_img), [], Nchannel);  % (X*Y*Z) x Nchannel
    mu_matrix = mean(reshaped, 1, 'omitnan');     % 1 x Nchannel
    sigma_matrix = std(reshaped, 0, 1, 'omitnan');

    for i=1:Npoint
        
        % Get voxel for each dot
        % current_point = allSpots.Centroid(i,:);
        current_point = table2array(allSpots(i, ["x", "y", "z"])); 
        extentsX = GetExtents(current_point(2), voxel_size(1), dimX);
        extentsY = GetExtents(current_point(1), voxel_size(2), dimY);                    
        extentsZ = GetExtents(current_point(3), voxel_size(3), dimZ);    

        current_voxel = double(input_img(extentsX, extentsY, extentsZ, :)); % 4-D array

        color_matrix(i, :) = single(squeeze(sum(current_voxel, [1 2 3]))); % sum along row,col,z
        color_matrix(i, :) = color_matrix(i, :) ./ (sqrt(sum(squeeze(color_matrix(i, :)).^2)) + 1E-6); % +1E-6 avoids denominator equaling to 0

        % z_voxel = (current_voxel - reshape(mu_matrix, [1 1 1 Nchannel])) ./ ...
        %       (reshape(sigma_matrix, [1 1 1 Nchannel]) + 1e-6);
        % z_voxel(z_voxel < 0) = 0; 
        % color_matrix(i, :) = squeeze(mean(z_voxel, [1 2 3], 'omitnan'));
        
        % for ch=1:4
        %     spot_region = current_voxel(:, :, :, ch);
        %     mu = mean(current_voxel(:,:,:,ch), 'omitnan');
        %     sigma = std(current_voxel(:,:,:,ch), 'omitnan');
        %     z_score = (spot_region - mu) ./ (sigma + eps);
        %     percentile_val = mean(normcdf(z_score), 'omitnan');
        %     color_matrix(i, ch) = percentile_val;
        % end

        % z_voxel = double(current_voxel);  % X x Y x Z x Nchannel
        % local_max_mask = false(size(z_voxel));
        % 
        % for ch = 1:Nchannel
        %     local_max_mask(:,:,:,ch) = imregionalmax(z_voxel(:,:,:,ch));
        % end
        % 
        % mean of local maxima
        % color_matrix(i,:) = squeeze(sum(z_voxel .* local_max_mask, [1 2 3])) ./ ...
        %                     max(sum(local_max_mask, [1 2 3]), 1);

        %disp(color_matrix(i,:));
        %determine if thers a sig in 1 channel
        color_max = max(color_matrix(i, :), [], 2);

        if ~isnan(color_max)
            m = find(color_matrix(i, :) == color_max);
            if numel(m) ~= 1
                color_seq(i) = "M";
                color_score(i) = Inf;
            else
                color_seq(i) = string(m(1));
                color_score(i) = -log(color_max);
            end
        else
            color_seq(i) = "N";
            color_score(i) = Inf;
        end
    end


end


function e = GetExtents(pos, voxelSize, lim)

if pos-voxelSize < 1 
    e1 = 1;
else
    e1 = pos-voxelSize;
end

if pos+voxelSize > lim
    e2 = lim;
else
    e2 = pos+voxelSize;
end

e = e1:e2;

end


function channel_scores = MaxFilterLocalMax(current_voxel, intensity_threshold)

    if nargin < 2
        intensity_threshold = 70; % default threshold
    end

    % Initialize output scores
    channel_scores = zeros(1, 4);

    vox_local = padarray(double(current_voxel), [7 7 4], 'replicate', 'both');
    for ch = 1:4
        ch_vol = current_voxel(:,:,:,ch);
        
        % Sum intensity for channel
        % total_intensity = mean(ch_vol(:));

        % 
        % % Check if local max exists: any voxel equals max in neighborhood & positive intensity
        % has_local_max = any(ch_vol(:) == max_filtered(:)) && max(ch_vol(:)) > 0;
        % 
        % % Override if total intensity passes threshold
        % if has_local_max || total_intensity > intensity_threshold
        %     channel_scores(ch) = total_intensity;
        % else
        %     channel_scores(ch) = 0;
        % end
        ch_local = vox_local(:,:,:,ch);
        max_val = max(ch_vol(:));
        min_val = min(ch_local(:));
        mean_val = mean(ch_local(:));
        
        if mean_val > intensity_threshold
            channel_scores(ch) = mean_val;
        else
            dyn_range = max_val / (min_val + 1E-6);
            channel_scores(ch) = dyn_range ^ 2;
        end

    end

end
