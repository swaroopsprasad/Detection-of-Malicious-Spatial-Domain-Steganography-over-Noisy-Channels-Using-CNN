function [lossy_cover_image, lossy_stego_image, packets_lost] = PacketLoss_Image_Generation(p, r, sub_mat_rows, sub_mat_cols, total_packets, sub_matrix_size, blockImage_cover, blockImage_stego)

%If p is the probability of transferring from Good State to the bad state
%and if r is the probability of transferring from the bad state to the Good
%state, given the p and r values, this code will generate a packet loss
%pattern (with burst losses) and save it to a file named 'Loss_Pattern.txt'

%% Generate Packet Loss pattern and store it in a text file
check = 100;
check_cnt = 1;
while check >= 10
good = 1;
packet_loss_pattern = [];
packet_size = 1;
while packet_size <= total_packets
if good == 1
    packet_loss_pattern = [packet_loss_pattern good]; %#ok<*AGROW>
    good = rand(1) > p;
elseif good == 0
    packet_loss_pattern = [packet_loss_pattern good];
    good = rand(1) > (1-r);
else
    fprintf('error\n');
    break;
end
packet_size = packet_size + 1;
end
file_name = strcat('Loss_Pattern_', num2str(total_packets), '.txt');
fid = fopen(file_name,'w');
fprintf(fid, '%d ', packet_loss_pattern);
fclose(fid);
received_packs = nnz(packet_loss_pattern);
theo_pack_loss_rate = 1 - r / (p+r);
act_pack_loss_rate = 1 - received_packs/total_packets;
check = abs(theo_pack_loss_rate - act_pack_loss_rate) / theo_pack_loss_rate * 100;
end
theo_pack_loss_rate = p / (p+r); %#ok<*NASGU>
act_pack_loss_rate = 1 - received_packs/total_packets;
% fprintf('Total Packets: ');disp(total_packets); 
% fprintf('Packets Received: '); disp(received_packs);
% fprintf('Packets lost: '); disp(total_packets - received_packs);
% fprintf('Packets Loss Rate (PLR): '); disp(act_pack_loss_rate);

%% Generate packet loss image with the help of packet loss pattern array 'packets'
lossy_cover_image = [];
lossy_stego_image = [];
packets_lost = 0;
row = 1;
col = 1;
random_stripe_type = 0;
for cnt=1:total_packets
    if(packet_loss_pattern(cnt) == 0)
            blockImage_cover(row,col) = {zeros(sub_matrix_size,sub_matrix_size,'uint8')};
            blockImage_stego(row,col) = {zeros(sub_matrix_size,sub_matrix_size,'uint8')};
            packets_lost = packets_lost + 1;
    end

    if(random_stripe_type > 0.5)
        row = row + 1;
        if (row > sub_mat_rows)
            col = col + 1;
            row = 1;
        end
        if (col > sub_mat_cols)
            break;
        end
    else
        col = col + 1;
        if (col > sub_mat_cols)
            row = row + 1;
            col = 1;
        end
        if (row > sub_mat_rows)
            break;
        end
    end
end
lossy_cover_image = cell2mat(blockImage_cover);
lossy_stego_image = cell2mat(blockImage_stego);
end