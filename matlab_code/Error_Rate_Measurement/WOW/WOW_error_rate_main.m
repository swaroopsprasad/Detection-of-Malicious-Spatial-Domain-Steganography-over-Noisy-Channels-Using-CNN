%           EXAMPLE - USING "WOW" embedding distortion
%
% -------------------------------------------------------------------------
% Copyright (c) 2012 DDE Lab, Binghamton University, NY.
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software for
% educational, research and non-profit purposes, without fee, and without a
% written agreement is hereby granted, provided that this copyright notice
% appears in all copies. The program is supplied "as is," without any
% accompanying services from DDE Lab. DDE Lab does not warrant the
% operation of the program will be uninterrupted or error-free. The
% end-user understands that the program was developed for research purposes
% and is advised not to rely exclusively on the program for any reason. In
% no event shall Binghamton University or DDE Lab be liable to any party
% for direct, indirect, special, incidental, or consequential damages,
% including lost profits, arising out of the use of this software. DDE Lab
% disclaims any warranties, and has no obligations to provide maintenance,
% support, updates, enhancements or modifications.
% -------------------------------------------------------------------------
% Author: Vojtech Holub
% -------------------------------------------------------------------------
% Contact: vojtech_holub@yahoo.com
%          fridrich@binghamton.edu
%          http://dde.binghamton.edu
% -------------------------------------------------------------------------
%% This program is a modified version for error rate detection
clc;
close all;
clear all;

fprintf('Embedding using Matlab file');
MEXstart = tic;

% Image paths
cover_source_path_BOSSBASE = 'H:\Masters Degree\Master_Thesis\Databases\BossBase_1.01_cover\';
cover_source_path_RAISE = 'H:\Masters Degree\Master_Thesis\Databases\Raise_dataset\cover_pgm\';

% Audit log Excel details
header_row = ["Image No", "Dataset", "Stego Type", "Payload", "Bit Count Actual", "Affected Type", "Standard Deviation", "Packet Size", "ABL", "PLR", "Total Number of Packets", "Number of Packets Lost", "PSNR", "Number of Bit Flips"];
write_rows = [];
write_rows = vertcat(header_row, write_rows);

% set params
params.p = -1;  % holder norm parameter

% Create sub matrcices
sub_matrix_sizes = [2, 4, 8, 16];

% Noise parameters 
std_dev_values = [0.3, 11];
peaksnr_LL = 30;
peaksnr_UL = 60;

% Iterate over a number of cover images
cnt = 1;
while(cnt <= 20)
    disp_cnt = 1;
    % set payload
    payload = 0.4;
    
    % load cover image
    if(cnt <= 10)
        cover_source_path = cover_source_path_BOSSBASE;
        dataset = "BOSSBASE";
    else
        cover_source_path = cover_source_path_RAISE;
        dataset = "RAISE";
    end
    coverPath = strcat(cover_source_path, strcat(num2str(cnt),'.pgm'));
    cover_image = imread(coverPath);
    [size_m, size_n] = size(cover_image);
    
    % Run embedding simulation
    [stego, bit_count, bit_positions] = WOW(cover_image, payload, params);
    
    stego_image = uint8(stego); 
    %cover_image = double(cover_image);
    
    %% Effect due to addition of Packet Loss
    while(disp_cnt <= 4)
        temp_row =[];
        temp_row = horzcat(temp_row, num2str(cnt)); %#ok<*AGROW>
        temp_row = horzcat(temp_row, dataset);
        temp_row = horzcat(temp_row, "WOW");
        temp_row = horzcat(temp_row, num2str(payload));
        temp_row = horzcat(temp_row, num2str(bit_count));
        
        % Add Packet loss to stego image
        temp_row = horzcat(temp_row, "Packet Loss");
        temp_row = horzcat(temp_row, "");
        
        peaksnr = 29;
        while(peaksnr < peaksnr_LL || peaksnr > peaksnr_UL)
            random_numb = disp_cnt;
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

            blockImage_stego = mat2cell(uint8(stego), sub_matrix_Length, sub_matrix_Length);

            total_packets = sub_mat_rows * sub_mat_cols;

            ABL = randi([4 30]);
            PLR = 0.001:0.001:0.01;
            PLR = PLR(randi([1,numel(PLR)]));
            prob_r = 1/ABL;
            prob_p = (prob_r*PLR)/(1-PLR);
            [lossy_cover_image, lossy_stego_image, packets_lost] = PacketLoss_Image_Generation(prob_p, prob_r, sub_mat_rows, sub_mat_cols, total_packets, sub_matrix_size, blockImage_cover, blockImage_stego);
            [peaksnr, ~] = psnr(lossy_cover_image, cover_image);
        end
        temp_row = horzcat(temp_row, num2str(sub_matrix_size));
        temp_row = horzcat(temp_row, num2str(ABL));
        temp_row = horzcat(temp_row, num2str(PLR));
        temp_row = horzcat(temp_row, num2str(total_packets));
        temp_row = horzcat(temp_row, num2str(packets_lost));
        
        temp_row = horzcat(temp_row, num2str(peaksnr));
        
        bit_flip_cnt = Calculate_Bit_Flips(cover_image, stego_image, lossy_stego_image, bit_positions);
        temp_row = horzcat(temp_row, num2str(bit_flip_cnt));
        
        write_rows = vertcat(write_rows, temp_row);
        
        %% Effect due to addition of Packet Loss and Noise
        disp_cnt_1 = 1;
        while(disp_cnt_1 <= length(std_dev_values))
            temp_row =[];
            temp_row = horzcat(temp_row, num2str(cnt)); %#ok<*AGROW>
            temp_row = horzcat(temp_row, dataset);
            temp_row = horzcat(temp_row, "WOW");
            temp_row = horzcat(temp_row, num2str(payload));
            temp_row = horzcat(temp_row, num2str(bit_count));

            temp_row = horzcat(temp_row, "Packet Loss + Gaussian Noise");
            std_dev = std_dev_values(disp_cnt_1);
            temp_row = horzcat(temp_row, num2str(std_dev));
            temp_row = horzcat(temp_row, num2str(sub_matrix_size));
            temp_row = horzcat(temp_row, num2str(ABL));
            temp_row = horzcat(temp_row, num2str(PLR));
            temp_row = horzcat(temp_row, num2str(total_packets));
            temp_row = horzcat(temp_row, num2str(packets_lost));
            
            noise_density = uint8(std_dev * randn(size_m,size_n));
            modified_stego_image = lossy_stego_image + noise_density;
            
            [peaksnr, ~] = psnr(modified_stego_image, cover_image);
            temp_row = horzcat(temp_row, num2str(peaksnr));
            bit_flip_cnt = Calculate_Bit_Flips(cover_image, stego_image, modified_stego_image, bit_positions);
            temp_row = horzcat(temp_row, num2str(bit_flip_cnt));

            write_rows = vertcat(write_rows, temp_row);
            disp_cnt_1 = disp_cnt_1 + 1;
        end
        disp_cnt = disp_cnt + 1;
    end
    
    disp_cnt = 1;
    %% Effect due to addition of Noise
    while(disp_cnt <= length(std_dev_values))
        temp_row =[];
        temp_row = horzcat(temp_row, num2str(cnt)); %#ok<*AGROW>
        temp_row = horzcat(temp_row, dataset);
        temp_row = horzcat(temp_row, "WOW");
        temp_row = horzcat(temp_row, num2str(payload));
        temp_row = horzcat(temp_row, num2str(bit_count));

        % Add noise to stego image
        temp_row = horzcat(temp_row, "Gaussian Noise");
        std_dev = std_dev_values(disp_cnt);
        temp_row = horzcat(temp_row, num2str(std_dev));
        temp_row = horzcat(temp_row, "");
        temp_row = horzcat(temp_row, "");
        temp_row = horzcat(temp_row, "");
        temp_row = horzcat(temp_row, "");
        temp_row = horzcat(temp_row, "");
        noise_density = uint8(std_dev * randn(size_m,size_n));
        modified_stego_image = stego_image + noise_density;
        
        [peaksnr, ~] = psnr(modified_stego_image, cover_image);
        temp_row = horzcat(temp_row, num2str(peaksnr));
        bit_flip_cnt = Calculate_Bit_Flips(cover_image, stego_image, modified_stego_image, bit_positions);
        temp_row = horzcat(temp_row, num2str(bit_flip_cnt));
        
        write_rows = vertcat(write_rows, temp_row);
        disp_cnt = disp_cnt + 1;
    end
    fprintf('\n Finished writing image %d \n', cnt);
    cnt = cnt + 1;
end

MEXend = toc(MEXstart);
% Store all required data into an excel
writematrix(write_rows, 'audit_log_Error_Rate_Detection_WOW_point_4.xlsx');
fprintf('Finished generating WOW-Error Rate Detection Excel\n');

function bit_flip_cnt = Calculate_Bit_Flips(cover, stego, modified_stego_image, bit_positions)
    % Code for checking amount of bit flips
    [size_m, size_n] = size(cover);
    bit_flip_cnt = 0;
    for m = 1:size_m
        for n = 1:size_n
            if(bit_positions(m,n) == 1)
                %cover_pixel = cover(m,n);
                stego_pixel = stego(m,n);
                modified_pixel = modified_stego_image(m,n);
                if(modified_pixel ~= stego_pixel)
%                     pixels_matters = bitget(bitxor(cover_pixel, stego_pixel),8:-1:1);
%                     positions = find(flip(pixels_matters) == 1);
                    if(~isequal(bitget(modified_pixel, 1), bitget(stego_pixel, 1)))
                        bit_flip_cnt = bit_flip_cnt + 1;
                    end
                end
            end
        end
    end
end