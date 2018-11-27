function [shuffled_data_plane_cell] = shuffle_single_cell_fn(bat_name,dates,data_file_name)
%% Loading bat details
% bat_name = 'Adren';
% dates = '181010';
% data_file_name = '1';

%% Parameters
TimeKeeper = tic;
data_folder_name = '/global/scratch/jelie/PlaneCells/Data_cluster';
% data_folder_name = 'D:\Yartsev lab\Height cell experiment\restructured data\';
bat_folder_name = fullfile(data_folder_name,bat_name);
bat_folder_name_date = strcat(bat_folder_name,filesep,dates);

nshuffle = 1000;
% Parameters of 3D rate map
frames_per_second = 180;
room_dimensions = [0 5400 0 5400 0 2000]; % x,y,z
X_room_dimensions = room_dimensions(1:2);
Y_room_dimensions = room_dimensions(3:4);
Z_room_dimensions = room_dimensions(5:6);
bin_size_pixels = 100;
mult_factor = 5;
min_distance_from_trajectory = 300; % in mm: This is the minimal distance we allow a voxel to be from
% the an "Active" voxel, i.e., a voxel which had some video time in it.
Min_time_in_voxel = 1; % in sec
smoothing_voxel_size_cm = 50;
bin_size_pixels_2_D = 200;
% Create gaussian kernel:
h_rate_map = 1.5; % This is what Hafting2005 defines as the smoothing factor which is actually
%the std of the gaussian kernel.
sigmaa = h_rate_map; % This the standard deviation of the kernel, which determines its width.
% for larger values, the kernel and is in fact wider. Our 1.5 bins
% kernel is relatively conservative in comparison to what people
% usually use.
hsize =4*round(sigmaa)+1; % hsize is the size of the kernel: I define it as 5*round(sigma)+1
% Just in order to make sure it is long enough to facilitate suffficent
% fading of the gaussian. In practise, it is much longer then what we
% actually need!
gaussian_kernel = fspecial('gaussian',hsize,sigmaa);
time_spent_minimum = 0.2 ; % Will discard pixels in which the animal spent < this minimal time (seconds)
spike_thresh = 40;

%% Configure paralle computing
if ~isempty(strfind(getenv('HOSTNAME'),'.savio')) || ~isempty(strfind(getenv('HOSTNAME'),'.brc'))
    fprintf('Initializing the cluster\n')
    parcluster 
    fprintf('This is the number fo available cores\n')
    feature('numCores')
    fprintf('This is the number of CPUs on the node\n')
    getenv('SLURM_CPUS_ON_NODE') 
    c = parcluster; 
    NumWorkNeeded = min(nshuffle,str2double(getenv('SLURM_CPUS_ON_NODE')));
    fprintf('We need %d workers/cores\n', NumWorkNeeded);
    c.numWorkers = NumWorkNeeded; 
    MyParPool = c.parpool(NumWorkNeeded,'IdleTimeout', Inf);
    system('mkdir -p /global/scratch/$USER/PlaneCells/$SLURM_JOB_ID')
    [~,JobID] = system('echo $SLURM_JOB_ID');
    parcluster.JobStorageLocation = ['/global/scratch/jelie/PlaneCells/' JobID];    
end
fprintf('End of parpool initalization\n')
toc(TimeKeeper)
%% Main loop
% cd(bat_folder_name_date)
load(fullfile(bat_folder_name_date,data_file_name))
fprintf('Done loading data\n')
toc(TimeKeeper)
shuffled_data_plane_cell = [];

PI_shuff_arr = NaN*ones(1,nshuffle);
SI3D_shuff_arr = NaN*ones(1,nshuffle);
SI_XY_shuff_arr = NaN*ones(1,nshuffle);
SI_ZY_shuff_arr = NaN*ones(1,nshuffle);
SI_XZ_shuff_arr = NaN*ones(1,nshuffle);
SI_prj_shuff_arr = NaN*ones(1,nshuffle);

x_video = data_plane_cell_single_cell.x_video_FE;
y_video = data_plane_cell_single_cell.y_video_FE;
z_video = data_plane_cell_single_cell.z_video_FE;
x_spikes = data_plane_cell_single_cell.x_spikes_FE;
y_spikes = data_plane_cell_single_cell.y_spikes_FE;
z_spikes = data_plane_cell_single_cell.z_spikes_FE;
pos = [x_video y_video z_video];
firpos_original = [x_spikes y_spikes z_spikes];

%Shuffling starts
parfor shuffle_index = 1:nshuffle
    
    fprintf('\n Shuffle number: %d/%d, Bat name: %s, Date: %s',shuffle_index,nshuffle,bat_name,dates)
    
    x_video_for_plotting = pos(:,1); y_video_for_plotting = pos(:,2); z_video_for_plotting = pos(:,3);
    ind_shuff = randsample(size(pos,1),size(firpos_original,1));
    x_spikes_shuffle = x_video_for_plotting(ind_shuff);
    y_spikes_shuffle = y_video_for_plotting(ind_shuff);
    z_spikes_shuffle = z_video_for_plotting(ind_shuff);
    firpos = [x_spikes_shuffle y_spikes_shuffle z_spikes_shuffle];
    
    % PI computation
    firpos2 = firpos/1000;
    pos2 = pos/1000;
    ptCloud = pointCloud(firpos2);
    ptCloudA = pcdenoise(ptCloud);
    firpos_trj_denoise = ptCloudA.Location;
    B_arr = nan(10,4);
    PI_arr = nan(10,1);
    for ii=1:10
        maxDistance=0.08*range(pos2(:,3));
        [model,inliers,outliers,err] = pcfitplane(ptCloudA,maxDistance);
        B_arr(ii,:) = model.Parameters;
        PI_arr(ii) = size(inliers,1)/size(firpos_trj_denoise,1);
    end
    [PI,index] = max(PI_arr);
    PI_shuff_arr(shuffle_index) = PI;
    B = B_arr(index,:);
    
    % 3D rate map
    x_spikes = firpos(:,1); y_spikes = firpos(:,2); z_spikes = firpos(:,3);
    x_spikes = x_spikes + 2600; y_spikes = y_spikes + 2600;
    x_video = pos(:,1); y_video = pos(:,2); z_video = pos(:,3);
    x_video = x_video + 2600; y_video = y_video + 2600;
    pos_behav_of_positive = [x_video y_video z_video];
    firpos_behav_of_positive = [x_spikes y_spikes z_spikes];
    
    mat_spike_density_raw = zeros( diff(Z_room_dimensions)/bin_size_pixels,...
        diff(Y_room_dimensions)/bin_size_pixels,diff(X_room_dimensions)/bin_size_pixels ) + NaN ; % Initialize
    mat_timespent_density_raw =  zeros( diff(Z_room_dimensions)/bin_size_pixels,...
        diff(Y_room_dimensions)/bin_size_pixels,diff(X_room_dimensions)/bin_size_pixels )  ; % Initialize
    
    place_field_density_raw =  zeros( diff(Z_room_dimensions)/bin_size_pixels,...
        diff(Y_room_dimensions)/bin_size_pixels,diff(X_room_dimensions)/bin_size_pixels ) + NaN ; % Initialize
    
    Spike_IXs_per_voxel = cell( diff(Z_room_dimensions)/bin_size_pixels,...
        diff(Y_room_dimensions)/bin_size_pixels,diff(X_room_dimensions)/bin_size_pixels ) ; % Initialize
    
    Video_IXs_per_voxel = cell( diff(Z_room_dimensions)/bin_size_pixels,...
        diff(Y_room_dimensions)/bin_size_pixels,diff(X_room_dimensions)/bin_size_pixels ) ; % Initialize
    
    for ii_x_bin = 1 : diff(X_room_dimensions)/bin_size_pixels , % Loop over x-bins
        for ii_y_bin = 1 : diff(Y_room_dimensions)/bin_size_pixels , % Loop over y-bins
            for ii_z_bin = 1 : diff(Z_room_dimensions)/bin_size_pixels , % Loop over z-bins
                % Spike Density:
                mat_spike_density_raw( ii_z_bin, ii_y_bin,ii_x_bin) = ... % All the data
                    sum( x_spikes >= 1 + bin_size_pixels*(ii_x_bin-1) & ...
                    x_spikes <  1 + bin_size_pixels*(ii_x_bin) & ...
                    y_spikes >= 1 + bin_size_pixels*(ii_y_bin-1) & ...
                    y_spikes <  1 + bin_size_pixels*(ii_y_bin) & ...
                    z_spikes >= 1 + bin_size_pixels*(ii_z_bin-1) & ...
                    z_spikes <  1 + bin_size_pixels*(ii_z_bin)) ;
                
                Spike_IXs_per_voxel{ii_z_bin, ii_y_bin, ii_x_bin} = find( x_spikes >= 1 + bin_size_pixels*(ii_x_bin-1) & ...
                    x_spikes <  1 + bin_size_pixels*(ii_x_bin) & ...
                    y_spikes >= 1 + bin_size_pixels*(ii_y_bin-1) & ...
                    y_spikes <  1 + bin_size_pixels*(ii_y_bin) & ...
                    z_spikes >= 1 + bin_size_pixels*(ii_z_bin-1) & ...
                    z_spikes<  1 + bin_size_pixels*(ii_z_bin)) ;
                
                
                % Time-Spent Density:
                mat_timespent_density_raw( ii_z_bin, ii_y_bin, ii_x_bin ) = ...
                    sum( x_video >= 1 + bin_size_pixels*(ii_x_bin-1) & ...
                    x_video <  1 + bin_size_pixels*(ii_x_bin) & ...
                    y_video >= 1 + bin_size_pixels*(ii_y_bin-1) & ...
                    y_video <  1 + bin_size_pixels*(ii_y_bin) & ...
                    z_video >= 1 + bin_size_pixels*(ii_z_bin-1) & ...
                    z_video <  1 + bin_size_pixels*(ii_z_bin) ) ;
                
                Video_IXs_per_voxel{ii_z_bin, ii_y_bin, ii_x_bin} = find( x_video >= 1 + bin_size_pixels*(ii_x_bin-1) & ...
                    x_video <  1 + bin_size_pixels*(ii_x_bin) & ...
                    y_video >= 1 + bin_size_pixels*(ii_y_bin-1) & ...
                    y_video <  1 + bin_size_pixels*(ii_y_bin) & ...
                    z_video >= 1 + bin_size_pixels*(ii_z_bin-1) & ...
                    z_video <  1 + bin_size_pixels*(ii_z_bin) ) ;
                
                % Normalize Time-Spent Density from Video-Frames-Spent to Seconds-Spent
                
                mat_timespent_density_raw( ii_z_bin, ii_y_bin, ii_x_bin ) = ...
                    mat_timespent_density_raw( ii_z_bin, ii_y_bin, ii_x_bin ) / frames_per_second ;
                
            end
        end
    end
    place_field_density_raw_with_NaN = zeros( diff(Z_room_dimensions)/bin_size_pixels,...
        diff(Y_room_dimensions)/bin_size_pixels,diff(X_room_dimensions)/bin_size_pixels ) + NaN ; % Initialize
    mat_time_density_raw_with_NaN = zeros( diff(Z_room_dimensions)/bin_size_pixels,...
        diff(Y_room_dimensions)/bin_size_pixels,diff(X_room_dimensions)/bin_size_pixels ) + NaN ; % Initialize
    mat_spike_count_raw_with_NaN = zeros( diff(Z_room_dimensions)/bin_size_pixels,...
        diff(Y_room_dimensions)/bin_size_pixels,diff(X_room_dimensions)/bin_size_pixels ) + NaN ; % Initialize
    
    % Create a MUCH larger "fictive" room into which our real room will go
    % into (this will be used later for the summation of the timespent).
    mat_timespent_density_raw_fictive = zeros( mult_factor*diff(Z_room_dimensions)/bin_size_pixels,...
        mult_factor*diff(Y_room_dimensions)/bin_size_pixels,mult_factor*diff(X_room_dimensions)/bin_size_pixels )  ; % Initialize
    mat_spike_density_raw_fictive = zeros( mult_factor*diff(Z_room_dimensions)/bin_size_pixels,...
        mult_factor*diff(Y_room_dimensions)/bin_size_pixels,mult_factor*diff(X_room_dimensions)/bin_size_pixels )  ; % Initialize
    
    % Note below the matrixs are of the REAL size as we will be
    % inserting the data directly into them:
    mat_associated_video_IXs = cell( diff(Z_room_dimensions)/bin_size_pixels,...
        diff(Y_room_dimensions)/bin_size_pixels,diff(X_room_dimensions)/bin_size_pixels ) ; % Initialize
    mat_associated_spike_IXs = cell( diff(Z_room_dimensions)/bin_size_pixels,...
        diff(Y_room_dimensions)/bin_size_pixels,diff(X_room_dimensions)/bin_size_pixels ) ; % Initialize
    
    
    real_Z_IXs = (mult_factor/2 - 0.5)*diff(Z_room_dimensions)/bin_size_pixels + 1:(mult_factor/2 + 0.5)*diff(Z_room_dimensions)/bin_size_pixels;
    real_Y_IXs = (mult_factor/2 - 0.5)*diff(Y_room_dimensions)/bin_size_pixels + 1:(mult_factor/2 + 0.5)*diff(Y_room_dimensions)/bin_size_pixels;
    real_X_IXs = (mult_factor/2 - 0.5)*diff(X_room_dimensions)/bin_size_pixels + 1:(mult_factor/2 + 0.5)*diff(X_room_dimensions)/bin_size_pixels;
    
    % insert our real maxtrix into the fictive one:
    mat_timespent_density_raw_fictive(real_Z_IXs,real_Y_IXs,real_X_IXs) = mat_timespent_density_raw;
    mat_spike_density_raw_fictive(real_Z_IXs,real_Y_IXs,real_X_IXs) = mat_spike_density_raw;
    
    % Now for each voxel we would like to compute the rate by using
    % adaptive smoothing implimented by increase the size of the voxel
    % until we have over a certain number of timespent in it (defined via
    % the variable: Min_time_in_voxel)
    validation_neighbors = min_distance_from_trajectory/bin_size_pixels;
    for ii_x_bin = 1 : diff(X_room_dimensions)/bin_size_pixels  , % Loop over x-bins
        for ii_y_bin = 1 : diff(Y_room_dimensions)/bin_size_pixels , % Loop over y-bins
            for ii_z_bin = 1 : diff(Z_room_dimensions)/bin_size_pixels , % Loop over z-bins
                current_voxel_timespent = mat_timespent_density_raw_fictive(real_Z_IXs(ii_z_bin),real_Y_IXs(ii_y_bin),real_X_IXs(ii_x_bin));
                neighbors_counter = 0;
                while (current_voxel_timespent<=Min_time_in_voxel)
                    neighbors_counter = neighbors_counter + 1; % Increase the voxel by one in each dimension
                    current_time_voxel = mat_timespent_density_raw_fictive(real_Z_IXs(ii_z_bin)-neighbors_counter:real_Z_IXs(ii_z_bin)+neighbors_counter,...
                        real_Y_IXs(ii_y_bin)-neighbors_counter:real_Y_IXs(ii_y_bin)+neighbors_counter,...
                        real_X_IXs(ii_x_bin)-neighbors_counter:real_X_IXs(ii_x_bin)+neighbors_counter);
                    current_voxel_timespent = sum(sum(sum(current_time_voxel)));
                    
                end
                
                current_voxel_IX_relative_to_original_mat =[ii_z_bin-neighbors_counter:ii_z_bin+neighbors_counter;...
                    ii_y_bin-neighbors_counter:ii_y_bin+neighbors_counter;...
                    ii_x_bin-neighbors_counter:ii_x_bin+neighbors_counter];
                
                
                % Check if this voxel is anywhere near an "active"
                % voxel
                current_time_valid = mat_timespent_density_raw_fictive(real_Z_IXs(ii_z_bin)-validation_neighbors:real_Z_IXs(ii_z_bin)+validation_neighbors,...
                    real_Y_IXs(ii_y_bin)-validation_neighbors:real_Y_IXs(ii_y_bin)+validation_neighbors,...
                    real_X_IXs(ii_x_bin)-validation_neighbors:real_X_IXs(ii_x_bin)+validation_neighbors);
                
                %                      current_time_valid_DBG = mat_timespent_density_raw(ii_z_bin-validation_neighbors:ii_z_bin,...
                %                         ii_y_bin-validation_neighbors:ii_y_bin,...
                %                         ii_x_bin-validation_neighbors:ii_x_bin);
                %
                %                     current_time_valid_DBG = mat_timespent_density_raw(ii_z_bin-validation_neighbors,...
                %                         ii_y_bin-validation_neighbors,...
                %                         ii_x_bin);
                
                current_spike_voxel = mat_spike_density_raw_fictive(real_Z_IXs(ii_z_bin)-neighbors_counter:real_Z_IXs(ii_z_bin)+neighbors_counter,...
                    real_Y_IXs(ii_y_bin)-neighbors_counter:real_Y_IXs(ii_y_bin)+neighbors_counter,...
                    real_X_IXs(ii_x_bin)-neighbors_counter:real_X_IXs(ii_x_bin)+neighbors_counter);
                
                current_voxel_spike_count = sum(sum(sum(current_spike_voxel)));
                
                if (sum(sum(sum(current_time_valid)))== 0) % i.e, there was NO activity in this distance from the central voxel (including it of course)
                    current_voxel_timespent = NaN; % we will not be using this voxel for analysis give it a vlaue of NaN here.
                    current_spike_voxel = NaN;
                else
                    % save the relevant position of animal + spike relevant
                    % for the computation in this voxel
                    for ii_x = current_voxel_IX_relative_to_original_mat(3,:)
                        for ii_y = current_voxel_IX_relative_to_original_mat(2,:)
                            for ii_z = current_voxel_IX_relative_to_original_mat(1,:)
                                current_IX_position = [ii_z,ii_y,ii_x];
                                % Check if Z IX is range
                                check_z = ((ii_z>0)&(ii_z<size(mat_associated_video_IXs,1)));
                                check_y = ((ii_y>0)&(ii_y<size(mat_associated_video_IXs,2)));
                                check_x = ((ii_x>0)&(ii_x<size(mat_associated_video_IXs,3)));
                                if (sum([check_z,check_y,check_x]) == 3)% meaning all are positive IXs
                                    mat_associated_video_IXs{ii_z_bin,ii_y_bin,ii_x_bin} = ...
                                        [mat_associated_video_IXs{ii_z_bin,ii_y_bin,ii_x_bin},...
                                        (Video_IXs_per_voxel{current_IX_position(1),...
                                        current_IX_position(2),current_IX_position(3)})'];
                                    
                                    mat_associated_spike_IXs{ii_z_bin,ii_y_bin,ii_x_bin} = ...
                                        [mat_associated_spike_IXs{ii_z_bin,ii_y_bin,ii_x_bin},...
                                        (Spike_IXs_per_voxel{current_IX_position(1),...
                                        current_IX_position(2),current_IX_position(3)})'];
                                else end
                                
                            end
                        end
                    end
                end
                place_field_density_raw_with_NaN(ii_z_bin,...
                    ii_y_bin,...
                    ii_x_bin) = current_voxel_spike_count/current_voxel_timespent;
                mat_time_density_raw_with_NaN(ii_z_bin,...
                    ii_y_bin,...
                    ii_x_bin) = current_voxel_timespent;
                mat_spike_count_raw_with_NaN(ii_z_bin,...
                    ii_y_bin,...
                    ii_x_bin)= current_voxel_spike_count;
                
            end
        end
    end
    
    transperancy_thres = 20; % in percentages of maximal firing rate
    transperancy_val_of_mx_FR = 4; % in percentages of maximal firing rate
    % along each dimension. The reason why we do this is to exclude cases where
    % a single voxel outlier would affect the field size estimation.
    X_room_dimensions_cm = room_dimensions(1:2)/10;
    Y_room_dimensions_cm = room_dimensions(3:4)/10;
    Z_room_dimensions_cm = room_dimensions(5:6)/10;
    
    
    max_FR = max(max(max(place_field_density_raw_with_NaN)));
    transperancy_thres_val = transperancy_thres*max_FR/100;
    value_set_at_below_thres_voxels = transperancy_val_of_mx_FR*max_FR/100;
    IXs = find(place_field_density_raw_with_NaN<transperancy_thres_val);
    place_field_density_raw_with_NaN_FE_adjusted = place_field_density_raw_with_NaN;
    if ~isempty(IXs)
        place_field_density_raw_with_NaN_FE_adjusted(IXs) = value_set_at_below_thres_voxels;
        place_field_density_raw_with_NaN_FE_adjusted(IXs(1))=0; % To set the scale from zero
    else end
    place_field_density_raw_with_NaN_FE_adjusted_reshaped = permute(place_field_density_raw_with_NaN_FE_adjusted,[2,3,1]);
    %D = squeeze(place_field_density_raw_with_NaN_FE_adjusted_reshaped);
    D = (place_field_density_raw_with_NaN_FE_adjusted_reshaped);
    
    
    idx_NaN_temp = find(isnan(place_field_density_raw_with_NaN));
    place_field_density_raw_FE_smoothed_no_NaN = place_field_density_raw_with_NaN;
    place_field_density_raw_FE_smoothed_no_NaN(idx_NaN_temp) = 0;
    mat_time_density_raw_FE_smoothed_no_NaN = mat_time_density_raw_with_NaN;
    mat_time_density_raw_FE_smoothed_no_NaN(idx_NaN_temp) = 0;
    % if rem(size(place_field_density_raw_FE_smoothed_no_NaN,1),2) || rem(size(place_field_density_raw_FE_smoothed_no_NaN,2),2) ...
    %         || rem(size(place_field_density_raw_FE_smoothed_no_NaN,3),2) == 0
    %     place_field_density_raw_FE_smoothed_no_NaN = place_field_density_raw_FE_smoothed_no_NaN(1:end-1,1:end-1,1:end-1);
    %     mat_time_density_raw_FE_smoothed_no_NaN = mat_time_density_raw_FE_smoothed_no_NaN(1:end-1,1:end-1,1:end-1);
    % end
    
    % We want to smooth the 3D matrix WITHOUT the NaNs:
    place_field_density_raw_with_NaN_FE_smoothed = smooth3(place_field_density_raw_FE_smoothed_no_NaN,...
        'gaussian',[smoothing_voxel_size_cm/10,smoothing_voxel_size_cm/10,smoothing_voxel_size_cm/10]);
    mat_time_density_raw_with_NaN_FE_smoothed = smooth3(mat_time_density_raw_FE_smoothed_no_NaN,...
        'gaussian',[smoothing_voxel_size_cm/10,smoothing_voxel_size_cm/10,smoothing_voxel_size_cm/10]);
    
    % Now return the NaNs so that we won't count the unvisited voxels:
    place_field_density_raw_with_NaN_FE_smoothed(idx_NaN_temp) = NaN;
    mat_time_density_raw_with_NaN_FE_smoothed(idx_NaN_temp) = NaN;
    
    
    % -----  INFORMATION PER SPIKE  (computed for the SMOOTHED field): -----
    % Information_per_spike = sum( p_i * ( r_i / r ) * log2( r_i / r ) )
    %    Where:
    %       r_i = firing rate in bin i ;
    %       p_i = occupancy of bin i = time-spent by bat in bin i / total time spent in all bins ;
    %       r = mean( r_i ) = overall mean firing rate (mean over all the pixels)
    % See: Skaggs WE, McNaughton BL, Wilson MA, Barnes CA, Hippocampus 6(2), 149-172 (1996).
    
    %find the not-NaN IXs;
    idx_notNaN_PlaceField_FE = find(~isnan(place_field_density_raw_with_NaN_FE_smoothed));
    % For the smoothed map:
    r_i = place_field_density_raw_with_NaN_FE_smoothed( idx_notNaN_PlaceField_FE ); % Use the SMOOTHED Place Field
    p_i = mat_time_density_raw_with_NaN_FE_smoothed( idx_notNaN_PlaceField_FE ) ./ ...
        sum( mat_time_density_raw_with_NaN_FE_smoothed( idx_notNaN_PlaceField_FE ) ) ;
    r_i = r_i(:) ; p_i = p_i(:) ; % Turn r_i and p_i into Column vectors
    % % %             r = mean( r_i ) ;
    r = sum( r_i .* p_i );
    information_per_spike_3_D_FE_smoothed = sum( p_i .* ( r_i / r ) .* log2( ( r_i + eps ) / r ) ) ; % I added a tiny number to avoid log(0)
    SI3D_shuff_arr(shuffle_index) = information_per_spike_3_D_FE_smoothed;
    
    %% 2D maps on canonical planes
    %XY
    mat_spike_density_raw_X_Y = zeros( diff(Y_room_dimensions)/bin_size_pixels_2_D,...
        diff(X_room_dimensions)/bin_size_pixels_2_D ) + NaN ; % Initialize
    mat_timespent_density_raw_X_Y =  zeros( diff(Y_room_dimensions)/bin_size_pixels_2_D,...
        diff(X_room_dimensions)/bin_size_pixels_2_D ) + NaN ; % Initialize
    place_field_density_raw_X_Y =  zeros( diff(Y_room_dimensions)/bin_size_pixels_2_D,...
        diff(X_room_dimensions)/bin_size_pixels_2_D ) + NaN ; % Initialize
    
    for ii_x_bin = 1 : diff(X_room_dimensions)/bin_size_pixels_2_D , % Loop over x-bins
        for ii_y_bin = 1 : diff(Y_room_dimensions)/bin_size_pixels_2_D , % Loop over y-bins
            % Spike Density:
            mat_spike_density_raw_X_Y( ii_y_bin,ii_x_bin) = ... % All the data
                sum( x_spikes >= 1 + bin_size_pixels_2_D*(ii_x_bin-1) & ...
                x_spikes <  1 + bin_size_pixels_2_D*(ii_x_bin) & ...
                y_spikes >= 1 + bin_size_pixels_2_D*(ii_y_bin-1) & ...
                y_spikes <  1 + bin_size_pixels_2_D*(ii_y_bin)) ;
            % Time-Spent Density:
            mat_timespent_density_raw_X_Y( ii_y_bin, ii_x_bin ) = ...
                sum( x_video >= 1 + bin_size_pixels_2_D*(ii_x_bin-1) & ...
                x_video <  1 + bin_size_pixels_2_D*(ii_x_bin) & ...
                y_video >= 1 + bin_size_pixels_2_D*(ii_y_bin-1) & ...
                y_video <  1 + bin_size_pixels_2_D*(ii_y_bin) ) ;
            % Normalize Time-Spent Density from Video-Frames-Spent to Seconds-Spent
            mat_timespent_density_raw_X_Y( ii_y_bin, ii_x_bin ) = ...
                mat_timespent_density_raw_X_Y( ii_y_bin, ii_x_bin ) / frames_per_second ;
            %             Discard pixels in which the animal Spent less than a certain Minimal amount of time --
            %             (this is computed for the "idx_include_VT" data only, usually resulting in
            % DIFFERENT pixels being discarded for the Full data):
            if ( mat_timespent_density_raw_X_Y( ii_y_bin, ii_x_bin) < time_spent_minimum ),
                mat_timespent_density_raw_X_Y( ii_y_bin, ii_x_bin) = 0 ; % Discard this time-spent-density pixel
                mat_timespent_density_raw_X_Y( ii_y_bin, ii_x_bin ) = 0 ; % Discard this spike-density pixel
                
            end
            if(mat_timespent_density_raw_X_Y~=0 & mat_timespent_density_raw_X_Y ==0)
                disp(2);
            end
        end
    end
    
    % Place Field = Spike Density / Time-Spent Density :
    %                     warning off MATLAB:divideByZero ;
    place_field_density_raw_X_Y = mat_spike_density_raw_X_Y ./ mat_timespent_density_raw_X_Y;
    %                     warning on MATLAB:divideByZero ;
    
    
    % Smoothing = convolve with gaussian kernel:
    mat_spike_density_smoothed_X_Y = imfilter(mat_spike_density_raw_X_Y,gaussian_kernel);
    mat_timespent_density_smoothed_X_Y = imfilter(mat_timespent_density_raw_X_Y,gaussian_kernel);
    
    % Place Field smoothed = Spike Density smoothed / Time-Spent Density smoothed :
    %                     warning off MATLAB:divideByZero ;
    place_field_density_smoothed_X_Y = mat_spike_density_smoothed_X_Y ./ mat_timespent_density_smoothed_X_Y ;
    %                     warning on MATLAB:divideByZero ;
    
    % ======= Compute the PF density with NaN's at unvisited location (will later be presented as white bins in the PF figure) : ==========
    
    % "Legalize" a bin (remove NaN) if the bat visited any of the bin's 8 closest neighbours:
    %                     warning off all
    idx_timespent_density = zeros( diff(Y_room_dimensions)/bin_size_pixels_2_D, diff(X_room_dimensions)/bin_size_pixels_2_D ) + NaN ; % Initialize
    for ii_x_bin = 2 : (diff(X_room_dimensions)/bin_size_pixels_2_D - 1) , % Loop over x-bins, NOT INCL. CAMERA-VIEW EDGES
        for ii_y_bin = 2 : (diff(Y_room_dimensions)/bin_size_pixels_2_D - 1) , % Loop over y-bins, NOT INCL. CAMERA-VIEW EDGES
            matrix_3x3_of_neighbors = ...
                mat_timespent_density_raw_X_Y( ii_y_bin-1 : ii_y_bin+1, ii_x_bin-1 : ii_x_bin+1 ) ;
            sum_including_the_central_bin = sum(sum( matrix_3x3_of_neighbors)); % Count the matrix_3x3_of_neighbors + the central bin itself
            if ( sum_including_the_central_bin  > 0 ), % If the animal visited any of this bin's 8 neighbors (3x3 region)
                idx_timespent_density(ii_y_bin,ii_x_bin) = 1; % Put 1 in the central bin
            else  % If the animal did NOT visit any of this bin's 8 neighbors (3x3 region)
                idx_timespent_density(ii_y_bin,ii_x_bin) = 0; % Put 0 in the central bin (later we will divide by this 0 and will get NaN for the firing-rate map)
            end
        end
    end
    %                     warning on all
    
    % Place Field = Spike Density / Time-Spent Density :
    %                     warning off MATLAB:divideByZero ;
    place_field_density_smoothed_with_NaN_X_Y = (place_field_density_smoothed_X_Y.* idx_timespent_density)./idx_timespent_density;
    mat_timespent_density_smoothed_with_NaN_X_Y = (mat_timespent_density_smoothed_X_Y.* idx_timespent_density)./idx_timespent_density;
    place_field_density_smoothed_with_NaN_normalized_X_Y = place_field_density_smoothed_with_NaN_X_Y./max(max(place_field_density_smoothed_with_NaN_X_Y));
    [r,c] = size(place_field_density_smoothed_with_NaN_X_Y);
    place_field_density_smoothed_with_NaN_5_level_binned_X_Y = ones(r,c)* NaN;
    for ii = 1:r
        for jj = 1:c
            if ~isnan(place_field_density_smoothed_with_NaN_X_Y(ii,jj))
                position = histc(place_field_density_smoothed_with_NaN_normalized_X_Y(ii,jj),[0:0.2:1]);
                place_field_density_smoothed_with_NaN_5_level_binned_X_Y(ii,jj) = sum(position(1:end).*[0.0:0.2:1]);
            else
            end
        end
    end
    
    %                     warning on MATLAB:divideByZero ;
    
    idx_notNaN_PlaceField_X_Y = find( ~isnan( place_field_density_smoothed_with_NaN_X_Y  ) ); % Find the indexes of non-NaN bins
    idx_isNaN_PlaceField_X_Y = find( isnan( place_field_density_smoothed_with_NaN_X_Y  ) ); % Find the indexes of NaN bins
    
    idx_notNaN_PlaceField_un_smoothed_rate_map_X_Y = find( ~isnan( place_field_density_raw_X_Y ) ); % Find the indexes of non-NaN bins
    
    peak_firing_rate_X_Y = max(max(place_field_density_smoothed_with_NaN_X_Y));
    
    
    % -----  INFORMATION PER SPIKE  (computed for the SMOOTHED field): -----
    % Information_per_spike = sum( p_i * ( r_i / r ) * log2( r_i / r ) )
    %    Where:
    %       r_i = firing rate in bin i ;
    %       p_i = occupancy of bin i = time-spent by bat in bin i / total time spent in all bins ;
    %       r = mean( r_i ) = overall mean firing rate (mean over all the pixels)
    % See: Skaggs WE, McNaughton BL, Wilson MA, Barnes CA, Hippocampus 6(2), 149-172 (1996).
    
    r_i = place_field_density_smoothed_with_NaN_X_Y( idx_notNaN_PlaceField_X_Y ); % Use the SMOOTHED Place Field
    p_i = mat_timespent_density_smoothed_with_NaN_X_Y( idx_notNaN_PlaceField_X_Y ) ./ ...
        sum( mat_timespent_density_smoothed_with_NaN_X_Y( idx_notNaN_PlaceField_X_Y ) ) ;
    r_i = r_i(:) ; p_i = p_i(:) ; % Turn r_i and p_i into Column vectors
    % % %             r = mean( r_i ) ;
    r = sum( r_i .* p_i );
    information_per_spike_X_Y = sum( p_i .* ( r_i / r ) .* log2( ( r_i + eps ) / r ) ) ; % I added a tiny number to avoid log(0)
    SI_XY = information_per_spike_X_Y;
    
    SI_XY_shuff_arr(shuffle_index) = SI_XY;
    %% ZY
    mat_spike_density_raw_Z_Y = zeros( diff(Z_room_dimensions)/bin_size_pixels_2_D,...
        diff(Y_room_dimensions)/bin_size_pixels_2_D ) + NaN ; % Initialize
    mat_timespent_density_raw_Z_Y =  zeros( diff(Z_room_dimensions)/bin_size_pixels_2_D,...
        diff(Y_room_dimensions)/bin_size_pixels_2_D ) + NaN ; % Initialize
    place_field_density_raw_Z_Y =  zeros( diff(Z_room_dimensions)/bin_size_pixels_2_D,...
        diff(Y_room_dimensions)/bin_size_pixels_2_D ) + NaN ; % Initialize
    
    for ii_Y_bin = 1 : diff(Y_room_dimensions)/bin_size_pixels_2_D , % Loop over Y-bins
        for ii_z_bin = 1 : diff(Z_room_dimensions)/bin_size_pixels_2_D , % Loop over z-bins
            % Spike Density:
            mat_spike_density_raw_Z_Y( ii_z_bin,ii_Y_bin) = ... % All the data
                sum( y_spikes >= 1 + bin_size_pixels_2_D*(ii_Y_bin-1) & ...
                y_spikes <  1 + bin_size_pixels_2_D*(ii_Y_bin) & ...
                z_spikes >= 1 + bin_size_pixels_2_D*(ii_z_bin-1) & ...
                z_spikes <  1 + bin_size_pixels_2_D*(ii_z_bin)) ;
            % Time-Spent Density:
            mat_timespent_density_raw_Z_Y( ii_z_bin, ii_Y_bin ) = ...
                sum( y_video >= 1 + bin_size_pixels_2_D*(ii_Y_bin-1) & ...
                y_video <  1 + bin_size_pixels_2_D*(ii_Y_bin) & ...
                z_video >= 1 + bin_size_pixels_2_D*(ii_z_bin-1) & ...
                z_video <  1 + bin_size_pixels_2_D*(ii_z_bin) ) ;
            % Normalize Time-Spent Density from Video-Frames-Spent to Seconds-Spent
            mat_timespent_density_raw_Z_Y( ii_z_bin, ii_Y_bin ) = ...
                mat_timespent_density_raw_Z_Y( ii_z_bin, ii_Y_bin ) / frames_per_second ;
            %             Discard pixels in which the animal Spent less than a certain Minimal amount of time --
            %             (this is computed for the "idx_include_VT" data only, usually resulting in
            % DIFFERENT pixels being discarded for the Full data):
            if ( mat_timespent_density_raw_Z_Y( ii_z_bin, ii_Y_bin) < time_spent_minimum ),
                mat_timespent_density_raw_Z_Y( ii_z_bin, ii_Y_bin) = 0 ; % Discard this time-spent-density pixel
                mat_timespent_density_raw_Z_Y( ii_z_bin, ii_Y_bin ) = 0 ; % Discard this spike-density pixel
                
            end
            if(mat_timespent_density_raw_Z_Y~=0 & mat_timespent_density_raw_Z_Y ==0)
                disp(2);
            end
        end
    end
    
    % Place Field = Spike Density / Time-Spent Density :
    %     warning off MATLAB:divideByZero ;
    place_field_density_raw_Z_Y = mat_spike_density_raw_Z_Y ./ mat_timespent_density_raw_Z_Y;
    %     warning on MATLAB:divideByZero ;
    
    
    % Smoothing = convolve with gaussian kernel:
    mat_spike_density_smoothed_Z_Y = imfilter(mat_spike_density_raw_Z_Y,gaussian_kernel);
    mat_timespent_density_smoothed_Z_Y = imfilter(mat_timespent_density_raw_Z_Y,gaussian_kernel);
    
    
    % Place Field smoothed = Spike Density smoothed / Time-Spent Density smoothed :
    %     warning off MATLAB:divideByZero ;
    place_field_density_smoothed_Z_Y = mat_spike_density_smoothed_Z_Y ./ mat_timespent_density_smoothed_Z_Y ;
    %     warning on MATLAB:divideByZero ;
    
    % ======= Compute the PF density with NaN's at unvisited location (will later be presented as white bins in the PF figure) : ==========
    
    % "Legalize" a bin (remove NaN) if the bat visited any of the bin's 8 closest neighbours:
    warning off all
    idx_timespent_density = zeros( diff(Z_room_dimensions)/bin_size_pixels_2_D, diff(Y_room_dimensions)/bin_size_pixels_2_D ) + NaN ; % Initialize
    for ii_Y_bin = 2 : (diff(Y_room_dimensions)/bin_size_pixels_2_D - 1) , % Loop over Y-bins, NOT INCL. CAMERA-VIEW EDGES
        for ii_z_bin = 2 : (diff(Z_room_dimensions)/bin_size_pixels_2_D - 1) , % Loop over z-bins, NOT INCL. CAMERA-VIEW EDGES
            matrix_3x3_of_neighbors = ...
                mat_timespent_density_raw_Z_Y( ii_z_bin-1 : ii_z_bin+1, ii_Y_bin-1 : ii_Y_bin+1 ) ;
            sum_including_the_central_bin = sum(sum( matrix_3x3_of_neighbors)); % Count the matrix_3x3_of_neighbors + the central bin itself
            if ( sum_including_the_central_bin  > 0 ), % If the animal visited any of this bin's 8 neighbors (3x3 region)
                idx_timespent_density(ii_z_bin,ii_Y_bin) = 1; % Put 1 in the central bin
            else  % If the animal did NOT visit any of this bin's 8 neighbors (3x3 region)
                idx_timespent_density(ii_z_bin,ii_Y_bin) = 0; % Put 0 in the central bin (later we will divide by this 0 and will get NaN for the firing-rate map)
            end
        end
    end
    warning on all
    
    % Place Field = Spike Density / Time-Spent Density :
    %     warning off MATLAB:divideByZero ;
    place_field_density_smoothed_with_NaN_Z_Y = (place_field_density_smoothed_Z_Y.* idx_timespent_density)./idx_timespent_density;
    mat_timespent_density_smoothed_with_NaN_Z_Y = (mat_timespent_density_smoothed_Z_Y.* idx_timespent_density)./idx_timespent_density;
    place_field_density_smoothed_with_NaN_normalized_Z_Y = place_field_density_smoothed_with_NaN_Z_Y./max(max(place_field_density_smoothed_with_NaN_Z_Y));
    [r,c] = size(place_field_density_smoothed_with_NaN_Z_Y);
    place_field_density_smoothed_with_NaN_5_level_binned_Z_Y = ones(r,c)* NaN;
    for ii = 1:r
        for jj = 1:c
            if ~isnan(place_field_density_smoothed_with_NaN_Z_Y(ii,jj))
                position = histc(place_field_density_smoothed_with_NaN_normalized_Z_Y(ii,jj),[0:0.2:1]);
                place_field_density_smoothed_with_NaN_5_level_binned_Z_Y(ii,jj) = sum(position(1:end).*[0.0:0.2:1]);
            else end
        end
    end
    
    %     warning on MATLAB:divideByZero ;
    
    idx_notNaN_PlaceField_Z_Y = find( ~isnan( place_field_density_smoothed_with_NaN_Z_Y  ) ); % Find the indexes of non-NaN bins
    idx_isNaN_PlaceField_Z_Y = find( isnan( place_field_density_smoothed_with_NaN_Z_Y  ) ); % Find the indexes of NaN bins
    
    idx_notNaN_PlaceField_un_smoothed_rate_map_Z_Y = find( ~isnan( place_field_density_raw_Z_Y ) ); % Find the indexes of non-NaN bins
    
    peak_firing_rate_Z_Y = max(max(place_field_density_smoothed_with_NaN_Z_Y));
    
    % -----  INFORMATION PER SPIKE  (computed for the SMOOTHED field): -----
    % Information_per_spike = sum( p_i * ( r_i / r ) * log2( r_i / r ) )
    %    Where:
    %       r_i = firing rate in bin i ;
    %       p_i = occupancy of bin i = time-spent by bat in bin i / total time spent in all bins ;
    %       r = mean( r_i ) = overall mean firing rate (mean over all the pixels)
    % See: Skaggs WE, McNaughton BL, Wilson MA, Barnes CA, Hippocampus 6(2), 149-172 (1996).
    
    r_i = place_field_density_smoothed_with_NaN_Z_Y( idx_notNaN_PlaceField_Z_Y ); % Use the SMOOTHED Place Field
    p_i = mat_timespent_density_smoothed_with_NaN_Z_Y( idx_notNaN_PlaceField_Z_Y ) ./ ...
        sum( mat_timespent_density_smoothed_with_NaN_Z_Y( idx_notNaN_PlaceField_Z_Y ) ) ;
    r_i = r_i(:) ; p_i = p_i(:) ; % Turn r_i and p_i into Column vectors
    % % %             r = mean( r_i ) ;
    r = sum( r_i .* p_i );
    information_per_spike_Z_Y = sum( p_i .* ( r_i / r ) .* log2( ( r_i + eps ) / r ) ) ; % I added a tiny number to avoid log(0)
    
    SI_ZY_shuff_arr(shuffle_index) = information_per_spike_Z_Y;
    
    %% XZ
    mat_spike_density_raw_X_Z = zeros( diff(Z_room_dimensions)/bin_size_pixels_2_D,...
        diff(X_room_dimensions)/bin_size_pixels_2_D ) + NaN ; % Initialize
    mat_timespent_density_raw_X_Z =  zeros( diff(Z_room_dimensions)/bin_size_pixels_2_D,...
        diff(X_room_dimensions)/bin_size_pixels_2_D ) + NaN ; % Initialize
    place_field_density_raw_X_Z =  zeros( diff(Z_room_dimensions)/bin_size_pixels_2_D,...
        diff(X_room_dimensions)/bin_size_pixels_2_D ) + NaN ; % Initialize
    
    for ii_x_bin = 1 : diff(X_room_dimensions)/bin_size_pixels_2_D , % Loop over x-bins
        for ii_z_bin = 1 : diff(Z_room_dimensions)/bin_size_pixels_2_D , % Loop over z-bins
            % Spike Density:
            mat_spike_density_raw_X_Z( ii_z_bin,ii_x_bin) = ... % All the data
                sum( x_spikes >= 1 + bin_size_pixels_2_D*(ii_x_bin-1) & ...
                x_spikes <  1 + bin_size_pixels_2_D*(ii_x_bin) & ...
                z_spikes >= 1 + bin_size_pixels_2_D*(ii_z_bin-1) & ...
                z_spikes <  1 + bin_size_pixels_2_D*(ii_z_bin)) ;
            % Time-Spent Density:
            mat_timespent_density_raw_X_Z( ii_z_bin, ii_x_bin ) = ...
                sum( x_video >= 1 + bin_size_pixels_2_D*(ii_x_bin-1) & ...
                x_video <  1 + bin_size_pixels_2_D*(ii_x_bin) & ...
                z_video >= 1 + bin_size_pixels_2_D*(ii_z_bin-1) & ...
                z_video <  1 + bin_size_pixels_2_D*(ii_z_bin) ) ;
            % Normalize Time-Spent Density from Video-Frames-Spent to Seconds-Spent
            mat_timespent_density_raw_X_Z( ii_z_bin, ii_x_bin ) = ...
                mat_timespent_density_raw_X_Z( ii_z_bin, ii_x_bin ) / frames_per_second ;
            %             Discard pixels in which the animal Spent less than a certain Minimal amount of time --
            %             (this is computed for the "idx_include_VT" data only, usually resulting in
            % DIFFERENT pixels being discarded for the Full data):
            if ( mat_timespent_density_raw_X_Z( ii_z_bin, ii_x_bin) < time_spent_minimum ),
                mat_timespent_density_raw_X_Z( ii_z_bin, ii_x_bin) = 0 ; % Discard this time-spent-density pixel
                mat_timespent_density_raw_X_Z( ii_z_bin, ii_x_bin ) = 0 ; % Discard this spike-density pixel
                
            end
            if(mat_timespent_density_raw_X_Z~=0 & mat_timespent_density_raw_X_Z ==0)
                disp(2);
            end
        end
    end
    
    % Place Field = Spike Density / Time-Spent Density :
    warning off MATLAB:divideByZero ;
    place_field_density_raw_X_Z = mat_spike_density_raw_X_Z ./ mat_timespent_density_raw_X_Z;
    warning on MATLAB:divideByZero ;
    
    
    % Smoothing = convolve with gaussian kernel:
    mat_spike_density_smoothed_X_Z = imfilter(mat_spike_density_raw_X_Z,gaussian_kernel);
    mat_timespent_density_smoothed_X_Z = imfilter(mat_timespent_density_raw_X_Z,gaussian_kernel);
    
    
    % Place Field smoothed = Spike Density smoothed / Time-Spent Density smoothed :
    warning off MATLAB:divideByZero ;
    place_field_density_smoothed_X_Z = mat_spike_density_smoothed_X_Z ./ mat_timespent_density_smoothed_X_Z ;
    warning on MATLAB:divideByZero ;
    
    % ======= Compute the PF density with NaN's at unvisited location (will later be presented as white bins in the PF figure) : ==========
    
    % "Legalize" a bin (remove NaN) if the bat visited any of the bin's 8 closest neighbours:
    warning off all
    idx_timespent_density = zeros( diff(Z_room_dimensions)/bin_size_pixels_2_D, diff(X_room_dimensions)/bin_size_pixels_2_D ) + NaN ; % Initialize
    for ii_x_bin = 2 : (diff(X_room_dimensions)/bin_size_pixels_2_D - 1) , % Loop over x-bins, NOT INCL. CAMERA-VIEW EDGES
        for ii_z_bin = 2 : (diff(Z_room_dimensions)/bin_size_pixels_2_D - 1) , % Loop over z-bins, NOT INCL. CAMERA-VIEW EDGES
            matrix_3x3_of_neighbors = ...
                mat_timespent_density_raw_X_Z( ii_z_bin-1 : ii_z_bin+1, ii_x_bin-1 : ii_x_bin+1 ) ;
            sum_including_the_central_bin = sum(sum( matrix_3x3_of_neighbors)); % Count the matrix_3x3_of_neighbors + the central bin itself
            if ( sum_including_the_central_bin  > 0 ), % If the animal visited any of this bin's 8 neighbors (3x3 region)
                idx_timespent_density(ii_z_bin,ii_x_bin) = 1; % Put 1 in the central bin
            else  % If the animal did NOT visit any of this bin's 8 neighbors (3x3 region)
                idx_timespent_density(ii_z_bin,ii_x_bin) = 0; % Put 0 in the central bin (later we will divide by this 0 and will get NaN for the firing-rate map)
            end
        end
    end
    warning on all
    
    % Place Field = Spike Density / Time-Spent Density :
    warning off MATLAB:divideByZero ;
    place_field_density_smoothed_with_NaN_X_Z = (place_field_density_smoothed_X_Z.* idx_timespent_density)./idx_timespent_density;
    mat_timespent_density_smoothed_with_NaN_X_Z = (mat_timespent_density_smoothed_X_Z.* idx_timespent_density)./idx_timespent_density;
    place_field_density_smoothed_with_NaN_normalized_X_Z = place_field_density_smoothed_with_NaN_X_Z./max(max(place_field_density_smoothed_with_NaN_X_Z));
    [r,c] = size(place_field_density_smoothed_with_NaN_X_Z);
    place_field_density_smoothed_with_NaN_5_level_binned_X_Z = ones(r,c)* NaN;
    for ii = 1:r
        for jj = 1:c
            if ~isnan(place_field_density_smoothed_with_NaN_X_Z(ii,jj))
                position = histc(place_field_density_smoothed_with_NaN_normalized_X_Z(ii,jj),[0:0.2:1]);
                place_field_density_smoothed_with_NaN_5_level_binned_X_Z(ii,jj) = sum(position(1:end).*[0.0:0.2:1]);
            else
            end
        end
    end
    
    warning on MATLAB:divideByZero ;
    
    idx_notNaN_PlaceField_X_Z = find( ~isnan( place_field_density_smoothed_with_NaN_X_Z  ) ); % Find the indexes of non-NaN bins
    idx_isNaN_PlaceField_X_Z = find( isnan( place_field_density_smoothed_with_NaN_X_Z  ) ); % Find the indexes of NaN bins
    
    idx_notNaN_PlaceField_un_smoothed_rate_map_X_Z = find( ~isnan( place_field_density_raw_X_Z ) ); % Find the indexes of non-NaN bins
    
    peak_firing_rate_X_Z = max(max(place_field_density_smoothed_with_NaN_X_Z));
    
    % -----  INFORMATION PER SPIKE  (computed for the SMOOTHED field): -----
    % Information_per_spike = sum( p_i * ( r_i / r ) * log2( r_i / r ) )
    %    Where:
    %       r_i = firing rate in bin i ;
    %       p_i = occupancy of bin i = time-spent by bat in bin i / total time spent in all bins ;
    %       r = mean( r_i ) = overall mean firing rate (mean over all the pixels)
    % See: Skaggs WE, McNaughton BL, Wilson MA, Barnes CA, Hippocampus 6(2), 149-172 (1996).
    
    r_i = place_field_density_smoothed_with_NaN_X_Z( idx_notNaN_PlaceField_X_Z ); % Use the SMOOTHED Place Field
    p_i = mat_timespent_density_smoothed_with_NaN_X_Z( idx_notNaN_PlaceField_X_Z ) ./ ...
        sum( mat_timespent_density_smoothed_with_NaN_X_Z( idx_notNaN_PlaceField_X_Z ) ) ;
    r_i = r_i(:) ; p_i = p_i(:) ; % Turn r_i and p_i into Column vectors
    % % %             r = mean( r_i ) ;
    r = sum( r_i .* p_i );
    information_per_spike_X_Z = sum( p_i .* ( r_i / r ) .* log2( ( r_i + eps ) / r ) ) ; % I added a tiny number to avoid log(0)
    
    SI_XZ_shuff_arr(shuffle_index) = information_per_spike_X_Z;
    
    %% 2D rate map (on projected plane)
    BB = B;
    N = (B(1:3)');
    % Projecting entire position to the fitted plane
    pos_behav_of_prj = nan(size(pos_behav_of_positive));
    for ii=1:size(pos_behav_of_positive,1)
        t = -(dot(N,pos_behav_of_positive(ii,:))+BB(4))/sum(N.^2);
        pos_behav_of_prj(ii,:) = pos_behav_of_positive(ii,:) + N'*t;
    end
    % Projecting firing field to the fitted plane
    firpos_behav_of_prj = nan(size(firpos_behav_of_positive));
    for ii=1:size(firpos_behav_of_positive,1)
        t = -(dot(N,firpos_behav_of_positive(ii,:))+BB(4))/sum(N.^2);
        firpos_behav_of_prj(ii,:) = firpos_behav_of_positive(ii,:) + N'*t;
    end
    pos_prj_var_ll_dim=var(pos_behav_of_prj);
    [~,index_sort] = sort(pos_prj_var_ll_dim,'descend');
    dim1 = index_sort(1); dim2 = index_sort(2);
    x_video_prj = pos_behav_of_prj(:,dim1);
    y_video_prj = pos_behav_of_prj(:,dim2);
    x_spikes_prj = firpos_behav_of_prj(:,dim1);
    y_spikes_prj = firpos_behav_of_prj(:,dim2);
    prj_size1 = round(max(pos_behav_of_prj(:,dim1))-min(pos_behav_of_prj(:,dim1)));
    prj_size2 = round(max(pos_behav_of_prj(:,dim2))-min(pos_behav_of_prj(:,dim2)));
    
    if rem(prj_size1,bin_size_pixels_2_D)~=0
        prj_size11 = [0 bin_size_pixels_2_D*ceil(prj_size1/bin_size_pixels_2_D)];
    else
        prj_size11 = [0 prj_size1];
    end
    if rem(prj_size2,bin_size_pixels_2_D)~=0
        prj_size22 = [0 bin_size_pixels_2_D*ceil(prj_size2/bin_size_pixels_2_D)];
    else
        prj_size22 = [0 prj_size2];
    end
    
    mat_spike_density_raw_prj = zeros( diff(prj_size22)/bin_size_pixels_2_D,...
        diff(prj_size11)/bin_size_pixels_2_D ) + NaN ; % Initialize
    mat_timespent_density_raw_prj =  zeros( diff(prj_size22)/bin_size_pixels_2_D,...
        diff(prj_size11)/bin_size_pixels_2_D ) + NaN ; % Initialize
    place_field_density_raw_prj =  zeros( diff(prj_size22)/bin_size_pixels_2_D,...
        diff(prj_size11)/bin_size_pixels_2_D ) + NaN ; % Initialize
    
    
    for ii_x_bin = 1 : diff(prj_size11)/bin_size_pixels_2_D  % Loop over x-bins
        for ii_z_bin = 1 : diff(prj_size22)/bin_size_pixels_2_D  % Loop over z-bins
            % Spike Density:
            mat_spike_density_raw_prj( ii_z_bin,ii_x_bin) = ... % All the data
                sum( x_spikes_prj >= 1 + bin_size_pixels_2_D*(ii_x_bin-1) & ...
                x_spikes_prj <  1 + bin_size_pixels_2_D*(ii_x_bin) & ...
                y_spikes_prj >= 1 + bin_size_pixels_2_D*(ii_z_bin-1) & ...
                y_spikes_prj <  1 + bin_size_pixels_2_D*(ii_z_bin)) ;
            % Time-Spent Density:
            mat_timespent_density_raw_prj( ii_z_bin, ii_x_bin ) = ...
                sum( x_video_prj >= 1 + bin_size_pixels_2_D*(ii_x_bin-1) & ...
                x_video_prj <  1 + bin_size_pixels_2_D*(ii_x_bin) & ...
                y_video_prj >= 1 + bin_size_pixels_2_D*(ii_z_bin-1) & ...
                y_video_prj <  1 + bin_size_pixels_2_D*(ii_z_bin) ) ;
            % Normalize Time-Spent Density from Video-Frames-Spent to Seconds-Spent
            mat_timespent_density_raw_prj( ii_z_bin, ii_x_bin ) = ...
                mat_timespent_density_raw_prj( ii_z_bin, ii_x_bin ) / frames_per_second ;
            %             Discard pixels in which the animal Spent less than a certain Minimal amount of time --
            %             (this is computed for the "idx_include_VT" data only, usually resulting in
            % DIFFERENT pixels being discarded for the Full data):
            if ( mat_timespent_density_raw_prj( ii_z_bin, ii_x_bin) < time_spent_minimum ),
                mat_timespent_density_raw_prj( ii_z_bin, ii_x_bin) = 0 ; % Discard this time-spent-density pixel
                mat_timespent_density_raw_prj( ii_z_bin, ii_x_bin ) = 0 ; % Discard this spike-density pixel
                
            end
            if(mat_timespent_density_raw_prj~=0 & mat_timespent_density_raw_prj ==0)
                disp(2);
            end
        end
    end
    
    
    % Place Field = Spike Density / Time-Spent Density :
    % warning off MATLAB:divideByZero ;
    place_field_density_raw_prj = mat_spike_density_raw_prj ./ mat_timespent_density_raw_prj;
    % warning on MATLAB:divideByZero ;
    
    
    % Smoothing = convolve with gaussian kernel:
    mat_spike_density_smoothed_prj = imfilter(mat_spike_density_raw_prj,gaussian_kernel);
    mat_timespent_density_smoothed_prj = imfilter(mat_timespent_density_raw_prj,gaussian_kernel);
    
    
    % Place Field smoothed = Spike Density smoothed / Time-Spent Density smoothed :
    % warning off MATLAB:divideByZero ;
    place_field_density_smoothed_prj = mat_spike_density_smoothed_prj ./ mat_timespent_density_smoothed_prj ;
    % warning on MATLAB:divideByZero ;
    
    % ======= Compute the PF density with NaN's at unvisited location (will later be presented as white bins in the PF figure) : ==========
    
    % "Legalize" a bin (remove NaN) if the bat visited any of the bin's 8 closest neighbours:
    % warning off all
    idx_timespent_density = zeros( diff(prj_size22)/bin_size_pixels_2_D, diff(prj_size11)/bin_size_pixels_2_D ) + NaN ; % Initialize
    for ii_x_bin = 2 : (diff(prj_size11)/bin_size_pixels_2_D - 1) , % Loop over x-bins, NOT INCL. CAMERA-VIEW EDGES
        for ii_z_bin = 2 : (diff(prj_size22)/bin_size_pixels_2_D - 1) , % Loop over z-bins, NOT INCL. CAMERA-VIEW EDGES
            matrix_3x3_of_neighbors = ...
                mat_timespent_density_raw_prj( ii_z_bin-1 : ii_z_bin+1, ii_x_bin-1 : ii_x_bin+1 ) ;
            sum_including_the_central_bin = sum(sum( matrix_3x3_of_neighbors)); % Count the matrix_3x3_of_neighbors + the central bin itself
            if ( sum_including_the_central_bin  > 0 ), % If the animal visited any of this bin's 8 neighbors (3x3 region)
                idx_timespent_density(ii_z_bin,ii_x_bin) = 1; % Put 1 in the central bin
            else  % If the animal did NOT visit any of this bin's 8 neighbors (3x3 region)
                idx_timespent_density(ii_z_bin,ii_x_bin) = 0; % Put 0 in the central bin (later we will divide by this 0 and will get NaN for the firing-rate map)
            end
        end
    end
    % warning on all
    
    % Place Field = Spike Density / Time-Spent Density :
    % warning off MATLAB:divideByZero ;
    place_field_density_smoothed_with_NaN_prj = (place_field_density_smoothed_prj.* idx_timespent_density)./idx_timespent_density;
    mat_timespent_density_smoothed_with_NaN_prj = (mat_timespent_density_smoothed_prj.* idx_timespent_density)./idx_timespent_density;
    place_field_density_smoothed_with_NaN_normalized_prj = place_field_density_smoothed_with_NaN_prj./max(max(place_field_density_smoothed_with_NaN_prj));
    [r,c] = size(place_field_density_smoothed_with_NaN_prj);
    place_field_density_smoothed_with_NaN_5_level_binned_prj = ones(r,c)* NaN;
    for ii = 1:r
        for jj = 1:c
            if ~isnan(place_field_density_smoothed_with_NaN_prj(ii,jj))
                position = histc(place_field_density_smoothed_with_NaN_normalized_prj(ii,jj),[0:0.2:1]);
                place_field_density_smoothed_with_NaN_5_level_binned_prj(ii,jj) = sum(position(1:end).*[0.0:0.2:1]);
            else
            end
        end
    end
    
    % warning on MATLAB:divideByZero ;
    
    idx_notNaN_PlaceField_prj = find( ~isnan( place_field_density_smoothed_with_NaN_prj  ) ); % Find the indexes of non-NaN bins
    idx_isNaN_PlaceField_prj = find( isnan( place_field_density_smoothed_with_NaN_prj ) ); % Find the indexes of NaN bins
    
    idx_notNaN_PlaceField_un_smoothed_rate_map_prj = find( ~isnan( place_field_density_raw_prj ) ); % Find the indexes of non-NaN bins
    
    peak_firing_rate_prj = max(max(place_field_density_smoothed_with_NaN_prj));
    
    % -----  INFORMATION PER SPIKE  (computed for the SMOOTHED field): -----
    % Information_per_spike = sum( p_i * ( r_i / r ) * log2( r_i / r ) )
    %    Where:
    %       r_i = firing rate in bin i ;
    %       p_i = occupancy of bin i = time-spent by bat in bin i / total time spent in all bins ;
    %       r = mean( r_i ) = overall mean firing rate (mean over all the pixels)
    % See: Skaggs WE, McNaughton BL, Wilson MA, Barnes CA, Hippocampus 6(2), 149-172 (1996).
    
    r_i = place_field_density_smoothed_with_NaN_prj( idx_notNaN_PlaceField_prj ); % Use the SMOOTHED Place Field
    p_i = mat_timespent_density_smoothed_with_NaN_prj( idx_notNaN_PlaceField_prj ) ./ ...
        sum( mat_timespent_density_smoothed_with_NaN_prj( idx_notNaN_PlaceField_prj ) ) ;
    r_i = r_i(:) ; p_i = p_i(:) ; % Turn r_i and p_i into Column vectors
    % % %             r = mean( r_i ) ;
    r = sum( r_i .* p_i );
    information_per_spike_prj = sum( p_i .* ( r_i / r ) .* log2( ( r_i + eps ) / r ) ) ; % I added a tiny number to avoid log(0)
    
    SI_prj_shuff_arr(shuffle_index) = information_per_spike_prj;
    
    
end

fprintf('Done running %d\n', nshuffle)
toc(TimeKeeper)

shuffled_data_plane_cell.SI3D = SI3D_shuff_arr;
shuffled_data_plane_cell.SIXY = SI_XY_shuff_arr;
shuffled_data_plane_cell.SIZY = SI_ZY_shuff_arr;
shuffled_data_plane_cell.SIXZ = SI_XZ_shuff_arr;
shuffled_data_plane_cell.SIprj = SI_prj_shuff_arr;
shuffled_data_plane_cell.PI = PI_shuff_arr;

shuffled_data_filename = strcat('shuffled_data_plane_cell_',bat_name,'_',dates,'_',data_file_name);
save(fullfile(bat_folder_name_date, shuffled_data_filename),'shuffled_data_plane_cell','-v7.3')
fprintf('Done saving data\n')
toc(TimeKeeper)
end
