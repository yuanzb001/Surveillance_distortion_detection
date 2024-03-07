clc;
clear;
close all;

data_folder = 'C:\Users\ikusanaa\PycharmProjects\IQA_Project\Testing_Dataset\Test_Data';
distortions = [1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 16, 17, 18];

dir_path = 'C:\Users\ikusanaa\PycharmProjects\IQA_Project\Testing_Dataset\';
distortions_name = ["gblur", "lblur", "mblur", "cdiff", "cshift", "csat1", "csat2", "wnoise", "wcnoise", "impnoise", "brit", "drkn", "mshift"];
levels = ["low", "mlow", "mid", "midh"', "high"];

list_folder = dir(data_folder);
lenlist = length(list_folder);

for i = 1:lenlist
    if(length(list_folder(i).name) > 2)
        file_path = fullfile(data_folder,list_folder(i).name);
        current_image = imread(file_path);
        [rows, columns, numberOfColorChannels] = size(current_image);
        filename = list_folder(i).name;
        if numberOfColorChannels > 1
            for n = 1 : length(distortions)
                for k = 1:5
                    distorted_img = imdist_generator(current_image, distortions(n), k);
                    write_path = strcat(dir_path, distortions_name(n),'\',levels(k),'\',filename);
                    disp(write_path);
                    imwrite(distorted_img, convertStringsToChars(write_path))

                end
            end
        else
            current_image = cat(3, current_image, current_image, current_image);
            for n = 1 : length(distortions)
                for k = 1:5
                    distorted_img = imdist_generator(current_image, distortions(n), k);
                    write_path = strcat(dir_path, distortions_name(n),'\',levels(k),'\',filename);
                    disp(write_path);
                    imwrite(distorted_img, convertStringsToChars(write_path))

                end
            end
            
        end
    end
end