%% This program will generate Packet loss images and stores relevant data into an excel
clc;
close all;
clear all;

% Image paths
cover_source_path_BOSSBASE = 'F:\Hiwi_Job\Databases\Alaska_cust\';
cover_source_path_RAISE = 'F:\Hiwi_Job\Databases\Alaska_cust\';

stego_source_path_BOSSBASE = 'F:\Hiwi_Job\Databases\Stego_Images\S-UNIWARD\dataset_point_4\';
stego_source_path_RAISE = 'F:\Hiwi_Job\Databases\Stego_Images\S-UNIWARD\dataset_point_4\';

destination_path_TRAINING = 'F:\Hiwi_Job\Spyder_Workspace\Packetloss_vs_Stego\Packetloss_vs_S_UNIWARD_4\dataset\training_set\';
destination_path_TEST = 'F:\Hiwi_Job\Spyder_Workspace\Packetloss_vs_Stego\Packetloss_vs_S_UNIWARD_4\dataset\test_set\';
destination_path_PREDICTION = 'F:\Hiwi_Job\Spyder_Workspace\Packetloss_vs_Stego\Packetloss_vs_S_UNIWARD_4\dataset\prediction_set\';

BOSSBASE_TRAINING = [1, 7000];
RAISE_TRAINING = [7001, 11000];

BOSSBASE_TEST = [11001, 14000];
RAISE_TEST = [14001, 15000];

RAISE_PREDICTION = [15001, 17000];

% Audit log Excel details
header_row = ["Image No", "Dataset", "Packet Size", "Average Burst Length(ABL)", "Packet Loss Rate (PLR)", "Prob_r", "Prob_p", "SNR", "PSNR"];

%% Generate Training data
write_rows = [];
write_rows = vertcat(header_row, write_rows);
disp_cnt = 1;

% BOSSBASE Training Section
cnt = BOSSBASE_TRAINING(1);
dataset = "BOSSBASE";
while(cnt <= BOSSBASE_TRAINING(2))
	temp_row =[];
	[temp_row] = generate_packetloss_stego_images(cnt, disp_cnt, cover_source_path_BOSSBASE, stego_source_path_BOSSBASE, temp_row, dataset, destination_path_TRAINING);
	write_rows = vertcat(write_rows, temp_row);
    fprintf('Finished writing image %d \n', cnt);
    cnt = cnt + 1;
	disp_cnt = disp_cnt + 1;
end

% RAISE Training Section
cnt = RAISE_TRAINING(1);
dataset = "RAISE";
while(cnt <= RAISE_TRAINING(2))
	temp_row =[];
	[temp_row] = generate_packetloss_stego_images(cnt, disp_cnt, cover_source_path_RAISE, stego_source_path_RAISE, temp_row, dataset, destination_path_TRAINING);
	write_rows = vertcat(write_rows, temp_row);
    fprintf('Finished writing image %d \n', cnt);
    cnt = cnt + 1;
	disp_cnt = disp_cnt + 1;
end

% Store all required data into an excel
writematrix(write_rows, 'audit_log_packetloss_training.xlsx');
fprintf('Finished generating Packetloss Training Images\n');

%% Generate Test data
write_rows = [];
write_rows = vertcat(header_row, write_rows);
disp_cnt = 1;

% BOSSBASE Test Section
cnt = BOSSBASE_TEST(1);
dataset = "BOSSBASE";
while(cnt <= BOSSBASE_TEST(2))
	temp_row =[];
	[temp_row] = generate_packetloss_stego_images(cnt, disp_cnt, cover_source_path_BOSSBASE, stego_source_path_BOSSBASE, temp_row, dataset, destination_path_TEST);
	write_rows = vertcat(write_rows, temp_row);
    fprintf('Finished writing image %d \n', cnt);
    cnt = cnt + 1;
	disp_cnt = disp_cnt + 1;
end

% RAISE Test Section
cnt = RAISE_TEST(1);
dataset = "RAISE";
while(cnt <= RAISE_TEST(2))
	temp_row =[];
	[temp_row] = generate_packetloss_stego_images(cnt, disp_cnt, cover_source_path_RAISE, stego_source_path_RAISE, temp_row, dataset, destination_path_TEST);
	write_rows = vertcat(write_rows, temp_row);
    fprintf('Finished writing image %d \n', cnt);
    cnt = cnt + 1;
	disp_cnt = disp_cnt + 1;
end

% Store all required data into an excel
writematrix(write_rows, 'audit_log_packetloss_test.xlsx');
fprintf('Finished generating Packetloss Test Images\n');

%% Generate Prediction data
write_rows = [];
write_rows = vertcat(header_row, write_rows);
disp_cnt = 1;

% RAISE Prediction Section
cnt = RAISE_PREDICTION(1);
dataset = "RAISE";
while(cnt <= RAISE_PREDICTION(2))
	temp_row =[];
	[temp_row] = generate_packetloss_stego_images(cnt, disp_cnt, cover_source_path_RAISE, stego_source_path_RAISE, temp_row, dataset, destination_path_PREDICTION);
	write_rows = vertcat(write_rows, temp_row);
    fprintf('Finished writing image %d \n', cnt);
    cnt = cnt + 1;
	disp_cnt = disp_cnt + 1;
end

% Store all required data into an excel
writematrix(write_rows, 'audit_log_packetloss_prediction.xlsx');
fprintf('Finished generating Packetloss Prediction Images\n');


%% Generate Packetloss Stego Images Function
function [temp_row] = generate_packetloss_stego_images(cnt, disp_cnt, cover_source_path, stego_source_path, temp_row, dataset, destination_path)
	generation_flag = false;
	
	% Create sub matrcices
	sub_matrix_sizes = [2, 4, 8, 16];

	% Get cover image 
    cover_path = strcat(cover_source_path, num2str(cnt), '.tif');
    cover_image = imread(cover_path);
    
    % Get stego image 
    stego_path = strcat(stego_source_path, num2str(cnt), '.bmp');
    stego_image = imread(stego_path);
    
    image_location_1 = strcat(destination_path, 'packetloss.', num2str(disp_cnt), '.bmp');
    image_location_2 = strcat(destination_path, 'stego.', num2str(disp_cnt), '.bmp');
	
	while(~generation_flag)
		random_numb = randi([1 4]);
		sub_matrix_size = sub_matrix_sizes(random_numb);    
		if (random_numb == 1)
			array_size = 256;           % 256 packets
		elseif (random_numb == 2)
			array_size = 128;           % 128 packets
		elseif (random_numb == 3)
			array_size = 64;            % 64 packets
		elseif (random_numb == 4)
			array_size = 32;            % 32 packets
		end
		sub_matrix_Length = sub_matrix_size * ones(1,array_size);
		blockImage_cover = mat2cell(cover_image, sub_matrix_Length, sub_matrix_Length);
		[sub_mat_rows , sub_mat_cols] = size(blockImage_cover);
		blockImage_stego = mat2cell(stego_image, sub_matrix_Length, sub_matrix_Length);
		
		% Generate different packet loss rates 
		% Find the probabilities 'p' and 'r' based on a random Average Burst Length
		% (ABL) between 3 and 30 and Packet Loss Ratio (PLR) between 0.1 to 0.5
		% p = (r*PLR)/(1-PLR); r = ((1-PLR)*p)/PLR; ABL = 1/r;

		total_packets = sub_mat_rows * sub_mat_cols;
		% pick a random ABL between 4 and 30
		ABL = randi([4 30]);
		% pick a random PLR between 0.001 and 0.01 (in 0.01 steps)
		PLR = 0.001:0.001:0.01;
		PLR = PLR(randi([1,numel(PLR)]));
		prob_r = 1/ABL;
		prob_p = (prob_r*PLR)/(1-PLR);
		[lossy_cover_image, lossy_stego_image, lossy_flag] = PacketLoss_Image_Generation(prob_p, prob_r, sub_mat_rows, sub_mat_cols, total_packets, sub_matrix_size, blockImage_cover, blockImage_stego);
		
		if(lossy_flag == 1)
			[peaksnr, snr] = psnr(lossy_cover_image, cover_image);

			% Save image only if PSNR in range of between 30 and 60
			if (peaksnr >= 30 && peaksnr <= 60)
				% save cover and packet loss images in their respective directories
				imwrite(lossy_cover_image, image_location_1);
				imwrite(lossy_stego_image, image_location_2);
				% Populate rows to write into excel
				temp_row = horzcat(temp_row, num2str(cnt)); %#ok<*AGROW>
				temp_row = horzcat(temp_row, dataset);
				temp_row = horzcat(temp_row, strcat(num2str(sub_matrix_size), ""));
				temp_row = horzcat(temp_row, strcat(num2str(ABL), ""));
				temp_row = horzcat(temp_row, num2str(PLR));
				temp_row = horzcat(temp_row, num2str(prob_r));
				temp_row = horzcat(temp_row, num2str(prob_p));
				temp_row = horzcat(temp_row, num2str(snr));
				temp_row = horzcat(temp_row, num2str(peaksnr));
			   
				generation_flag = true;
			end
		end
	end
end