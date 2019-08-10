function classifyResult = challenge(recordName, dataset_dir)
%
% Sample entry for the 2017 PhysioNet/CinC Challenge.
%
% INPUTS:
% recordName: string specifying the record name to process
%
% OUTPUTS:
% classifyResult: integer value where
%                     N = normal rhythm
%                     A = AF
%                     O = other rhythm
%                     ~ = noisy recording (poor signal quality)
%
% To run your entry on the entire training set in a format that is
% compatible with PhysioNet's scoring enviroment, run the script
% generateValidationSet.m
%
% The challenge function requires that you have downloaded the challenge
% data 'training_set' in a subdirectory of the current directory.
%    http://physionet.org/physiobank/database/challenge/2017/
%
% This dataset is used by the generateValidationSet.m script to create
% the annotations on your training set that will be used to verify that
% your entry works properly in the PhysioNet testing environment.
%
%
% Version 1.0
%
% Written by: Chengyu Liu and Qiao Li January 20 2017
%             chengyu.liu@emory.edu  qiao.li@emory.edu
%
% Modified by: 
% Shreyasi Datta
% Chetanya Puri
% Ayan Mukherjee
% Rohan Banerjee
% Anirban Dutta Choudhury
% Arijit Ukil
% Soma Bandyopadhyay
% Rituraj Singh
% Arpan Pal
% Sundeep Khandelwal

% classifyResult = 'N'; % default output normal rhythm
% read the filename
[tm,ecg,fs,siginfo]=rdmat(recordName);

% Cancel noise and get clean ECG
raw_ecg = ecg;
%[ ecg ] = ecg_noisecancellation( ecg, fs );

% Compute the ECG points
[ P_index, Q_index, R_index, S_index, T_index] = ecg_points( ecg, fs );

% ECG features
[Features_SD] = ECG_features_158_old(ecg, fs, P_index, Q_index, R_index, S_index, T_index);
if length(Features_SD) == 1
    Features_SD = zeros(1,68);
end
Features_RB = other_features_new(ecg, fs);
%Features_RB = [Features_RB frequency_features(ecg, fs)];
Features_ADC = pr( R_index, P_index);
feat_ind = 1:27;
Features_CP = extract_features(raw_ecg, fs, feat_ind);
Features_embcsoa = soa_features(ecg, fs, R_index);
features_temp = new_feat_sd_158_old(ecg, fs, R_index);
features_temp_rs = generate_30_features(ecg,fs); 

f1 = [Features_SD Features_RB Features_ADC Features_embcsoa Features_CP features_temp features_temp_rs];

% Need to add NaN and Inf handling
load MeanVector_208
MeanVect = MeanVector_208;
replaceFeat = union(find(isnan(f1)),find(isinf(abs(f1))));
f1(replaceFeat) = MeanVect(replaceFeat);
%Extract the file name
fStr = string(extractBetween(recordName,strcat(dataset_dir,"/"),length(recordName)))
%disp(fStr)
save(fStr, 'f1')
% load train_model
load BoostS1 
load BoostS2 
load BoostS3

%{
label = CascadedClassifier2LayerUpload(f1,BoostS1,BoostS2,BoostS3);

% back to original labels
if (label==0)
    classifyResult = 'N';
elseif(label==1)
    classifyResult = 'A';
elseif(label==2)
    classifyResult = 'O';
else
    classifyResult = '~';
end
disp(classifyResult)
end
%}
classifyResult = 'N'
