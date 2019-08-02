% This script will verify that your code is working as you intended, by
% running it on a small subset (300 records) of the training data, then
% comparing the answers.txt file that you submit with your entry with
% answers produced by your code running in our test environment using
% the same records.
%
% In order to run this script, you should have downloaded and extracted
% the validation set into the directory containing this file.
%
%
% Written by: Chengyu Liu and Qiao Li January 20 2017 
%             chengyu.liu@emory.edu  qiao.li@emory.edu
%
% Last modified by:
%
%
function generateValidationSet(model_dir_name, dataset_dir)

%clear all;
%close all;
%clc
data_dir = [pwd filesep 'validation' filesep 'data' filesep model_dir_name filesep dataset_dir filesep];

%% Add this directory to the MATLAB path.
addpath(pwd)

%% Check for previous files before starting validation procedure
answers = dir(['answers.txt']);
if(~isempty(answers))
    while(1)
        display(['Found previous answer sheet file in: ' pwd])
        cont = upper(input('Delete it (Y/N/Q)?','s'));
        if(strcmp(cont,'Y') || strcmp(cont,'N') || strcmp(cont,'Q'))
            if(~strcmp(cont,'Y'))
                display('Exiting script!!')
                return;
            end
            break;
        end
    end
    display('Removing previous answer sheet.')
    delete(answers.name);
end
display(data_dir)
%% Load the list of records in the validation set.
fid = fopen([data_dir 'RECORDS'],'r');
if(fid ~= -1)
    RECLIST = textscan(fid,'%s');
else
    error(['Could not open ' data_dir 'RECORDS for scoring. Exiting...'])
end
fclose(fid);
RECORDS = RECLIST{1};

%% Running on the validation set and obtain the score results
classifyResult = zeros(length(RECORDS),1);
total_time     = 0;

fid=fopen('answers.txt','wt');
for i = 1:length(RECORDS)
    fname = RECORDS{i};
    tic;
    classifyResult(i) = challenge([data_dir fname], dataset_dir);

    % write the answer to answers.txt file
    fprintf(fid,'%s,%s\n',RECORDS{i},classifyResult(i));

    total_time = total_time+toc;
    fprintf(['---Processed ' num2str(i) ' out of ' num2str(length(RECORDS)) ' records.\n'])
end
fclose(fid);

