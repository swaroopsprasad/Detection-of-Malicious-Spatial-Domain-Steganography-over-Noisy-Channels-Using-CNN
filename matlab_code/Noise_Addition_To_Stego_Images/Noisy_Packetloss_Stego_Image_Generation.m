%% This program will generate different types of noisy images and stores all required data into an excel sheet
clc;
clear all;
close all;

% Image paths
cover_source_path_TRAINING = 'F:\Hiwi_Job\Spyder_Workspace\Packetloss_vs_Stego\Packetloss_vs_S_UNIWARD_4\dataset\training_set\packetloss.';
cover_source_path_TEST = 'F:\Hiwi_Job\Spyder_Workspace\Packetloss_vs_Stego\Packetloss_vs_S_UNIWARD_4\dataset\test_set\packetloss.';
cover_source_path_PREDICTION = 'F:\Hiwi_Job\Spyder_Workspace\Packetloss_vs_Stego\Packetloss_vs_S_UNIWARD_4\dataset\prediction_set\packetloss.';

stego_source_path_TRAINING = 'F:\Hiwi_Job\Spyder_Workspace\Packetloss_vs_Stego\Packetloss_vs_S_UNIWARD_4\dataset\training_set\stego.';
stego_source_path_TEST = 'F:\Hiwi_Job\Spyder_Workspace\Packetloss_vs_Stego\Packetloss_vs_S_UNIWARD_4\dataset\test_set\stego.';
stego_source_path_PREDICTION = 'F:\Hiwi_Job\Spyder_Workspace\Packetloss_vs_Stego\Packetloss_vs_S_UNIWARD_4\dataset\prediction_set\stego.';

destination_path_TRAINING = 'F:\Hiwi_Job\Spyder_Workspace\Packetloss_Noisy_vs_Stego\Packetloss_Noisy_vs_S_UNIWARD_4\dataset\training_set\';
destination_path_TEST = 'F:\Hiwi_Job\Spyder_Workspace\Packetloss_Noisy_vs_Stego\Packetloss_Noisy_vs_S_UNIWARD_4\dataset\test_set\';
destination_path_PREDICTION = 'F:\Hiwi_Job\Spyder_Workspace\Packetloss_Noisy_vs_Stego\Packetloss_Noisy_vs_S_UNIWARD_4\dataset\prediction_set\';


TRAINING_LIMITS = [1, 11000];
TEST_LIMITS = [1, 4000];
PREDICTION_LIMITS = [1, 2000];

% Audit log Excel details
header_row = ["Image No", "Noise Type", "Noise Density", "SNR", "PSNR"];

%% Generate Training data
write_rows = [];
write_rows = vertcat(header_row, write_rows);
disp_cnt = 1;

% Training Section
cnt = TRAINING_LIMITS(1);
while(cnt <= TRAINING_LIMITS(2))
	temp_row =[];
	[temp_row] = generate_noisy_stego_images(cnt, disp_cnt, cover_source_path_TRAINING, stego_source_path_TRAINING, temp_row, destination_path_TRAINING);
	% write rows in excel sheet
	write_rows = vertcat(write_rows, temp_row);
	fprintf('Finished writing image %d \n', cnt);
	cnt = cnt + 1;
	disp_cnt = disp_cnt + 1;
end

% Store all required data into an excel
writematrix(write_rows, 'audit_log_noisyloss_training.xlsx');
fprintf('Finished generating Noisy Training Images\n');



%% Generate Test data
write_rows = [];
write_rows = vertcat(header_row, write_rows);
disp_cnt = 1;

% Testing Section
cnt = TEST_LIMITS(1);
while(cnt <= TEST_LIMITS(2))
	temp_row =[];
	[temp_row] = generate_noisy_stego_images(cnt, disp_cnt, cover_source_path_TEST, stego_source_path_TEST, temp_row, destination_path_TEST);
	% write rows in excel sheet
	write_rows = vertcat(write_rows, temp_row);
	fprintf('Finished writing image %d \n', cnt);
	cnt = cnt + 1;
	disp_cnt = disp_cnt + 1;
end

% Store all required data into an excel
writematrix(write_rows, 'audit_log_noisyloss_test.xlsx');
fprintf('Finished generating Noisy Test Images\n');

%% Generate Prediction data
write_rows = [];
write_rows = vertcat(header_row, write_rows);
disp_cnt = 1;

% Prediction Section
cnt = PREDICTION_LIMITS(1);
while(cnt <= PREDICTION_LIMITS(2))
	temp_row =[];
	[temp_row] = generate_noisy_stego_images(cnt, disp_cnt, cover_source_path_PREDICTION, stego_source_path_PREDICTION, temp_row, destination_path_PREDICTION);
	% write rows in excel sheet
	write_rows = vertcat(write_rows, temp_row);
	fprintf('Finished writing image %d \n', cnt);
	cnt = cnt + 1;
	disp_cnt = disp_cnt + 1;
end

% Store all required data into an excel
writematrix(write_rows, 'audit_log_noisyloss_prediction.xlsx');
fprintf('Finished generating Noisy Prediction Images\n');

%% Generate Noisy Stego Images Function
function [temp_row] = generate_noisy_stego_images(cnt, disp_cnt, cover_source_path, stego_source_path, temp_row, destination_path)
	generation_flag = false;
	% Get cover image
	cover_path = strcat(cover_source_path, num2str(cnt), '.bmp');
	cover_image = imread(cover_path);
	
	% Get stego image
	stego_path = strcat(stego_source_path, num2str(cnt), '.bmp');
	stego_image = imread(stego_path);
	
	[size_m , size_n] = size(cover_image);
	
	image_location_1 = strcat(destination_path, 'noisyloss.', num2str(disp_cnt), '.bmp');
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
            temp_row = horzcat(temp_row, "Gaussian");
            temp_row = horzcat(temp_row, num2str(std_dev));
            temp_row = horzcat(temp_row, num2str(snr));
			temp_row = horzcat(temp_row, num2str(peaksnr));
			
			generation_flag = true;
		end
	end
end