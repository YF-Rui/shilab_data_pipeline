function props = SpotFindingMax3D( input_img, intensity_estimation, intensity_threshold )
%SpotFindingMax3D 

    props = [];
    Nchannel = size(input_img, 4);
    
    for c=1:Nchannel
        current_channel = input_img(:,:,:,c);
        current_max = imextendedmax(current_channel, max);

        switch intensity_estimation
            case "adaptive"
                max_intensity = max(current_channel, [], 'all');
                current_threshold = max_intensity * intensity_threshold;
                if class(current_channel) == "uint8"
                    current_threshold = max(current_threshold, 0.1 * 255);
                elseif class(current_channel) == "uint16"
                    current_threshold = max(current_threshold, 0.1 * 65535);
                else
                    error("Unsupported image type");
                end
            case "global"
                if class(current_channel) == "uint8"
                    current_threshold = intensity_threshold * 255;
                elseif class(current_channel) == "uint16"
                    current_threshold = intensity_threshold * 65535;
                else
                    error("Unsupported image type");
                end
        end
        current_output = current_max & current_channel > current_threshold;

        current_props = regionprops3(current_output, current_channel, ["Centroid", "MaxIntensity"]);
        current_props.Centroid = int16(current_props.Centroid);
        current_props.Channel = repmat(c, size(current_props, 1), 1);
        props = vertcat(props, current_props);
    end
    if ~isempty(current_props)
        figure;
        scatter3(current_props.Centroid(:,1), current_props.Centroid(:,2), current_props.Centroid(:,3), ...
            50, 'filled'); % 50 is marker size
        title(['Centroids for channel ', num2str(c)]);
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
        grid on;
        axis equal;
    end

end

