% run registration and spot finding workflow 
% user need to define the location of the dataet configuration file
% config_path 
function sdata = rsf_workflow_culture(config_path)

    % load dataset info 
    config_raw = fileread(config_path);
    config = jsondecode(config_raw);
    setenv('KMP_DUPLICATE_LIB_OK','TRUE');
    %pyenv('Version', 'C:/Users/xgao76/AppData/Local/anaconda3/envs/spotiflow/python.exe');
    % config.output_path = strcat(config.output_path(1:end-1), "_noeq");
    % add path for .m files
    addpath(fullfile(pwd, './code-base/src/')) % pwd is the location of the starfinder folder

    % iterate through each fov
    % for n=config.starting_fov_id:config.starting_fov_id + config.number_of_fovs - 1
    for i = 1:length(config.fov_id_list)
    
        well_id = config.fov_well_list{i};
        n = config.fov_id_list(i);
        current_fov = sprintf(config.fov_id_pattern, n, well_id);

        % create object instance
        sdata = STARMapDataset(config.input_path, config.output_path, 'useGPU', true);
        
        % create log folder and file path
        log_folder = fullfile(config.output_path, "log");
        if ~exist(log_folder, 'dir')
            mkdir(log_folder);
        end
        diary_file = fullfile(log_folder, sprintf("%s.txt", current_fov));
        if exist(diary_file, 'file'); delete(diary_file); end
        diary(diary_file);

        starting = tic;
        disp("Current FOV: " + current_fov);

        % load sequencing images 
        sdata = sdata.LoadRawImages('fovID', current_fov, 'rotate_angle', config.rotate_angle);
        sdata.layers.ref = config.ref_round;

        % load additional images
        sdata = sdata.LoadRawImages('fovID', current_fov, ... 
                                    'rotate_angle', config.rotate_angle, ...
                                    'folder_list', string(config.additional_round), ...
                                    'channel_order_dict', config.channel_order, ...
                                    'update_layer_slot', "other");

        % preprocessing
        sdata = sdata.EnhanceContrast("min-max");
        sdata = sdata.EnhanceContrast("min-max", 'layer', sdata.layers.other);
        sdata = sdata.HistEqualize('reference_channel',4);

        %SaveEqualizedImages(sdata, 'F:/equalized_tiffs_to638');

        % registration
        sdata = sdata.GlobalRegistration;

        % save reference images 
        ref_merged_folder = fullfile(config.output_path, "images", "ref_merged");
        if ~exist(ref_merged_folder, 'dir')
            mkdir(ref_merged_folder);
        end
        ref_merged_fname = fullfile(ref_merged_folder, sprintf('%s.tif', current_fov));
        if config.maximum_projection
            SaveSingleStack(max(sdata.registration{sdata.layers.ref}, [], 3), ref_merged_fname);
        else
            SaveSingleStack(sdata.registration{sdata.layers.ref}, ref_merged_fname);
        end

        % load reference nuclei image for additional registration 
        refernce_dapi_fname = dir(fullfile(config.input_path, 'round1', current_fov, '*405*.tif'));
        current_ref_img = LoadMultipageTiff(fullfile(refernce_dapi_fname.folder, refernce_dapi_fname.name), false);
        current_ref_img = imrotate(current_ref_img, config.rotate_angle);
        sdata = sdata.GlobalRegistration('layer', sdata.layers.other, ...
                                        'ref_img', 'input_image_ref', ...
                                        'input_image_mov', current_ref_img, ...
                                        'mov_img', 'single-channel', ...
                                        'ref_channel', config.ref_channel);
        % local registration (optional)                                    
        sdata = sdata.LocalRegistration;

        % spot finding s
        sdata = sdata.SpotFinding('method',"max3d",'intensity_threshold', 0.2);%"spotiflow", 'probability_threshold', 0.01, 'intensity_threshold', 0.1);
        sdata = sdata.ReadsExtraction('voxel_size', [3 3 1]);
        sdata = sdata.LoadCodebook;
        sdata = sdata.ReadsFiltration('save_scores', true, 'save_qscore', true, 'end_base', config.end_base);

        % output 
        sdata = sdata.MakeProjection;
        preview_folder = fullfile(config.output_path, "images", "montage_preview");
        if ~exist(preview_folder, 'dir')
            mkdir(preview_folder);
        end
        % projection_preview_path = fullfile(preview_folder, sprintf("%s.tif", current_fov));
        % sdata = sdata.ViewProjection('save', true, 'output_path', projection_preview_path);

        sdata = sdata.SaveImages('layer', sdata.layers.other, 'output_path', config.output_path, 'folder_format', "single", 'maximum_projection', config.maximum_projection);
        %ref_merge_max = max(sdata.registration{sdata.layers.ref}, [], 3);
        %sdata = sdata.ViewSignal('bg_img', ref_merge_max, 'save', true);
        sdata = sdata.SaveSignal;

        toc(starting);
        diary off;

        clearvars('-except', 'config', 'i');  % keep config and loop counter
        for g = 1:2
            reset(gpuDevice(g));  % free GPU memory
        end
        java.lang.System.gc; 

    end

end

