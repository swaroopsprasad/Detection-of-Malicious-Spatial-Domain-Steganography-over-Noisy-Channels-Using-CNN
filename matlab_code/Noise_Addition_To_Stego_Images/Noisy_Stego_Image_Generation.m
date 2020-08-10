%% This program will generate different types of noisy images and stores all required data into an excel sheet
clc;
clear all;
close all;

% Image paths
cover_source_path_BOSSBASE = 'I:\Hiwi_Job\Databases\Alaska_cust\';
cover_source_path_RAISE = 'I:\Hiwi_Job\Databases\Alaska_cust\';

stego_source_path_BOSSBASE = 'I:\Hiwi_Job\Databases\Stego_Images\S-UNIWARD\dataset_point_4\';
stego_source_path_RAISE = 'I:\Hiwi_Job\Databases\Stego_Images\S-UNIWARD\dataset_point_4\';

destination_path_TRAINING = 'I:\Hiwi_Job\Spyder_Workspace\Noisy_vs_Stego\Noisy_vs_S_UNIWARD_4\dataset\training_set\';
destination_path_TEST = 'I:\Hiwi_Job\Spyder_Workspace\Noisy_vs_Stego\Noisy_vs_S_UNIWARD_4\dataset\test_set\';
destination_path_PREDICTION = 'I:\Hiwi_Job\Spyder_Workspace\Noisy_vs_Stego\Noisy_vs_S_UNIWARD_4\dataset\prediction_set\';

BOSSBASE_TRAINING = [1, 7000];
RAISE_TRAINING = [7001, 11000];

BOSSBASE_TEST = [11001, 14000];
RAISE_TEST = [14001, 15000];

RAISE_PREDICTION = [15001, 17000];

% Audit log Excel details
header_row = ["Image No", "Dataset", "Noise Type", "Noise Density", "SNR", "PSNR"];

%% Generate Training data
write_rows = [];
write_rows = vertcat(header_row, write_rows);
disp_cnt = 1;

% BOSSBASE Training Section
cnt = BOSSBASE_TRAINING(1);
dataset = "BOSSBASE";
while(cnt <= BOSSBASE_TRAINING(2))
	temp_row =[];
	[temp_row] = generate_noisy_stego_images(cnt, disp_cnt, cover_source_path_BOSSBASE, stego_source_path_BOSSBASE, temp_row, dataset, destination_path_TRAINING);
	% write rows in excel sheet
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
	[temp_row] = generate_noisy_stego_images(cnt, disp_cnt, cover_source_path_RAISE, stego_source_path_RAISE, temp_row, dataset, destination_path_TRAINING);
	% write rows in excel sheet
	write_rows = vertcat(write_rows, temp_row);
	fprintf('Finished writing image %d \n', cnt);
	cnt = cnt + 1;
	disp_cnt = disp_cnt + 1;
end

% Store all required data into an excel
writematrix(write_rows, 'audit_log_noisy_training.xlsx');
fprintf('Finished generating Noisy Training Images\n');



%% Generate Test data
write_rows = [];
write_rows = vertcat(header_row, write_rows);
disp_cnt = 1;

% BOSSBASE Test Section
cnt = BOSSBASE_TEST(1);
dataset = "BOSSBASE";
while(cnt <= BOSSBASE_TEST(2))
	temp_row =[];
	[temp_row] = generate_noisy_stego_images(cnt, disp_cnt, cover_source_path_BOSSBASE, stego_source_path_BOSSBASE, temp_row, dataset, destination_path_TEST);
	% write rows in excel sheet
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
	[temp_row] = generate_noisy_stego_images(cnt, disp_cnt, cover_source_path_RAISE, stego_source_path_RAISE, temp_row, dataset, destination_path_TEST);
	% write rows in excel sheet
	write_rows = vertcat(write_rows, temp_row);
	fprintf('Finished writing image %d \n', cnt);
	cnt = cnt + 1;
	disp_cnt = disp_cnt + 1;
end

% Store all required data into an excel
writematrix(write_rows, 'audit_log_noisy_test.xlsx');
fprintf('Finished generating Noisy Test Images\n');

%% Generate Prediction data
write_rows = [];
write_rows = vertcat(header_row, write_rows);
destination_path = destination_path_PREDICTION;
disp_cnt = 1;

% RAISE Prediction Section
cnt = RAISE_PREDICTION(1);
dataset = "RAISE";
while(cnt <= RAISE_PREDICTION(2))
	temp_row =[];
	[temp_row] = generate_noisy_stego_images(cnt, disp_cnt, cover_source_path_RAISE, stego_source_path_RAISE, temp_row, dataset, destination_path_PREDICTION);
	% write rows in excel sheet
	write_rows = vertcat(write_rows, temp_row);
	fprintf('Finished writing image %d \n', cnt);
	cnt = cnt + 1;
	disp_cnt = disp_cnt + 1;
end

% Store all required data into an excel
writematrix(write_rows, 'audit_log_noisy_prediction.xlsx');
fprintf('Finished generating Noisy Prediction Images\n');

%% Generate Noisy Stego Images Function
function [temp_row] = generate_noisy_stego_images(cnt, disp_cnt, cover_source_path, stego_source_path, temp_row, dataset, destination_path)
	generation_flag = false;
	% Get cover image
	cover_path = strcat(cover_source_path, num2str(cnt), '.tif');
	cover_image = imread(cover_path);
	
	% Get stego image
	stego_path = strcat(stego_source_path, num2str(cnt), '.bmp');
	stego_image = imread(stego_path);
	
	[size_m , size_n] = size(cover_image);
	
	image_location_1 = strcat(destination_path, 'noisy.', num2str(disp_cnt), '.bmp');
	image_location_2 = strcat(destination_path, 'stego.', num2str(disp_cnt), '.bmp');
	while(~generation_flag)
		rand_numb = rand(1);
		% Add Gaussian noise to the image 
		if (rand_numb >= 0.5)
			std_dev = randi([1 11]);                        % Std. Dev between 1 and 11
		else
			std_dev = randi([3 9])/10;                      % Std. Dev between 0.3 and 0.9
		end                                       
		noise_density = uint8(std_dev * randn(size_m,size_n));
		noisy_image = cover_image + noise_density;
		noisy_stego_image = stego_image + noise_density;
		
		%% Calculate SNR and PSNR
		[peaksnr, snr] = psnr(noisy_image, cover_image);
		if (peaksnr >= 30 && peaksnr <= 60)
			% save images to destination paths
			imwrite(noisy_image, image_location_1);
			imwrite(noisy_stego_image, image_location_2);
			
			temp_row = horzcat(temp_row, num2str(cnt)); %#ok<*AGROW>
            temp_row = horzcat(temp_row, dataset); %#ok<*AGROW>
            temp_row = horzcat(temp_row, "Gaussian");
            temp_row = horzcat(temp_row, num2str(std_dev));
            temp_row = horzcat(temp_row, num2str(snr));
			temp_row = horzcat(temp_row, num2str(peaksnr));
			
			generation_flag = true;
		end
	end
end