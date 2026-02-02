function props = SpotFindingSpotiflow(input_img, probability_threshold, intensity_threshold)

    props = [];
    Nchannel = size(input_img, 4);
    
    np = py.importlib.import_module('numpy');
    spotiflow = py.importlib.import_module('spotiflow.model');
    model = spotiflow.Spotiflow.from_pretrained('smfish_3d');
    spot_data_all = table();
    for c=1:Nchannel
        current_channel = input_img(:,:,:,c);
        
        % Convert to numpy array (float32)
        img_py = np.array(single(current_channel));
        
        % Predict with Spotiflow mode
        output = predict(model, img_py, pyargs('subpix', false, 'prob_thresh', probability_threshold, "verbose", false, 'device',"cuda"));

        points = output{1}; 
        details = output{2}; 
        intens = details.intens; 
        prob = details.prob;

        intens = double(py.numpy.array(intens).flatten());
        prob = double(py.numpy.array(prob).flatten());
        points = points.tolist();
        nSpots = length(points);
        points_ml = zeros(nSpots, 3);
        for i = 1:nSpots
            coords = points{i};      
            points_ml(i,1) = double(coords{2});
            points_ml(i,2) = double(coords{1});
            points_ml(i,3) = double(coords{3});
        end
        intens = reshape(intens, [], 1);
        prob = reshape(prob, [], 1);


        % % Apply intensity threshold
        keep_idx = logical(prob > probability_threshold);
        points_ml = points_ml(keep_idx, :);
        intens = intens(keep_idx);
        prob = prob(keep_idx);

        max_intensity = max(current_channel, [], 'all');
        current_threshold = max_intensity * intensity_threshold;
        keep_idx = logical(intens > current_threshold);
        points_ml = points_ml(keep_idx, :);
        intens = intens(keep_idx);
        prob = prob(keep_idx);
        
        % Round to int16 for centroids
        centroids = int16(round(points_ml));
        nSpots = size(centroids, 1);
        
        % Create output table with Centroid, Intensity, Channel
        T = table;
        T.Centroid = centroids;
        T.MaxIntensity = intens;
        T.Channel = repmat(c, nSpots, 1);
        T.prob = prob;
        props = vertcat(props, T);

    end


end
