clc; clear; close all;

%% ============================================
% SETTINGS (YOUR PARAMETERS)
%% ============================================
folder = 'C:\Users\rudra\Downloads\Paderborn';
files = dir(fullfile(folder,'*.mat'));

fs = 64000;          % sampling frequency
mechFreq = 4000;     % not used here but kept
decFactor = 32;       % 64k → 2k
fs_new = fs / decFactor;

win = 128;            % same time span as 2048@64k
overlap = 64;



step = win - overlap;

EPS = 1e-8;

allFeatures = [];
allLabels = [];

%% ============================================
% LOOP FILES
%% ============================================
for f = 1:length(files)

    filePath = fullfile(folder,files(f).name);
    data = load(filePath);

    fn = fieldnames(data);
    s = data.(fn{1});

    % ----- Y vibration -----
   y = double(s.Y(1).Data(:));
y = y - mean(y);

% decimate
y = decimate(y, decFactor);

% bandpass 10–300
[b,a] = butter(4,[10 300]/(fs_new/2),'bandpass');
y = filtfilt(b,a,y);

% normalize
y = (y - mean(y)) / (std(y)+EPS);


% ----- BANDPASS 10–300 Hz -----
lowcut = 10;
highcut = 300;

[b,a] = butter(4,[lowcut highcut]/(fs_new/2),'bandpass');
y = filtfilt(b,a,y);

    [b,a] = butter(4,[lowcut highcut]/(fs/2),'bandpass');
    y = filtfilt(b,a,y);

    % ----- NORMALIZE -----
    y = (y - mean(y)) / (std(y)+EPS);

    % ----- LABEL -----
    name = files(f).name;
    if contains(name,'K001')
        label = 0;
    elseif contains(name,'KA')
        label = 1;
    elseif contains(name,'KI')
        label = 2;
    else
        continue
    end

    % ----- WINDOW COUNT -----
    numSeg = floor((length(y)-win)/step) + 1;
    if numSeg <= 0
        continue
    end

    %% ============================================
    % WINDOW LOOP
    %% ============================================
    for i = 1:numSeg

        idx = (i-1)*step + (1:win);
        x = y(idx);

        %% ===== PYTHON TIME FEATURES =====
        len = numel(x);
        Mean = mean(x);
        RMS = sqrt(mean(x.^2));
        Std = std(x);
        Var = var(x);
        P2P = max(x) - min(x);
        Kurt = kurtosis(x(:));
        Skew = skewness(x(:));
        Energy = sum(x.^2);

        mean_abs = mean(abs(x)) + EPS;
        max_abs = max(abs(x)) + EPS;
        sqrt_mean_abs = mean(sqrt(abs(x)+EPS));

        Shape = RMS / mean_abs;
        Crest = max_abs / (RMS + EPS);
        Impulse = max_abs / mean_abs;
        Margin = max_abs / (sqrt_mean_abs^2 + EPS);

        counts = histcounts(x,64,'Normalization','probability');
        counts = counts + EPS;
        Entropy = -sum(counts .* log(counts));

        dx = diff(x);
        ddx = diff(dx);

        Mobility = sqrt(var(dx)/(var(x)+EPS));
        Complexity = sqrt(var(ddx)/(var(dx)+EPS)) / (Mobility+EPS);

        feat = [len Mean RMS Std Var P2P Kurt Skew Energy ...
                Shape Crest Impulse Margin Entropy Mobility Complexity];

        allFeatures = [allFeatures; feat];
        allLabels = [allLabels; label];

    end
end

%% ============================================
% SAVE CSV
%% ============================================
dataset = [allFeatures allLabels];

headers = {'length','mean','rms','std','var','peak_to_peak',...
           'kurtosis','skewness','energy',...
           'shape_factor','crest_factor','impulse_factor','margin_factor',...
           'entropy','hjorth_mobility','hjorth_complexity','label'};

headers = headers(1:size(dataset,2));

T = array2table(dataset,'VariableNames',headers);

csvFile = fullfile(pwd,'paderborn_time_features_python_style.csv');
writetable(T,csvFile);

disp('✅ Saved: paderborn_time_features_python_style.csv');
fprintf('Samples: %d   Features: %d\n',size(allFeatures,1),size(allFeatures,2));

clc; clear; close all;

%% ============================================
% LOAD TRAIN / TEST DATA (FROM YOUR FEATURE CODE)
%% ============================================
trainTable = readtable('paderborn_time_features_python_style.csv');
testTable  = readtable('paderborn_time_features_python_style.csv');

Xtrain = trainTable{:,1:end-1};
Ytrain = trainTable.label;

Xtest  = testTable{:,1:end-1};
Ytest  = testTable.label;

fprintf('Train shape: %d samples, %d features\n',size(Xtrain,1),size(Xtrain,2));
fprintf('Test shape : %d samples, %d features\n',size(Xtest,1),size(Xtest,2));

classNames = {'NORMAL','INNER','OUTER','BALL','COMBINED'};

results = struct();

%% ============================================
% 1️⃣ KNN MODEL
%% ============================================
disp('==================================================')
disp('Training KNN')
disp('==================================================')

knnModel = fitcknn(Xtrain,Ytrain,...
    'NumNeighbors',7,...
    'Distance','euclidean',...
    'Standardize',false);

Ypred_knn = predict(knnModel,Xtest);

acc_knn = mean(Ypred_knn==Ytest);
prec_knn = precision_macro(Ytest,Ypred_knn);
rec_knn  = recall_macro(Ytest,Ypred_knn);
f1_knn   = f1_macro(Ytest,Ypred_knn);

fprintf('Accuracy : %.2f %%\n',acc_knn*100);
fprintf('Precision: %.2f %%\n',prec_knn*100);
fprintf('Recall   : %.2f %%\n',rec_knn*100);
fprintf('F1-score : %.2f %%\n',f1_knn*100);

figure;
confusionchart(Ytest,Ypred_knn,'RowSummary','row-normalized');
title('KNN Confusion Matrix');

results.KNN = acc_knn;

%% ============================================
% 2️⃣ SVM (RBF)
%% ============================================
disp('==================================================')
disp('Training SVM (RBF)')
disp('==================================================')

template = templateSVM('KernelFunction','rbf','KernelScale','auto');

svmModel = fitcecoc(Xtrain,Ytrain,...
    'Learners',template,...
    'Coding','onevsone');

Ypred_svm = predict(svmModel,Xtest);

acc_svm = mean(Ypred_svm==Ytest);
prec_svm = precision_macro(Ytest,Ypred_svm);
rec_svm  = recall_macro(Ytest,Ypred_svm);
f1_svm   = f1_macro(Ytest,Ypred_svm);

fprintf('Accuracy : %.2f %%\n',acc_svm*100);
fprintf('Precision: %.2f %%\n',prec_svm*100);
fprintf('Recall   : %.2f %%\n',rec_svm*100);
fprintf('F1-score : %.2f %%\n',f1_svm*100);

figure;
confusionchart(Ytest,Ypred_svm,'RowSummary','row-normalized');
title('SVM (RBF) Confusion Matrix');

results.SVM = acc_svm;

%% ============================================
% SUMMARY
%% ============================================
disp('==================================================')
disp('FINAL ACCURACY SUMMARY')
disp('==================================================')

fprintf('KNN : %.2f %%\n',results.KNN*100);
fprintf('SVM : %.2f %%\n',results.SVM*100);

%% ============================================
% METRIC FUNCTIONS
%% ============================================
function p = precision_macro(ytrue, ypred)
    classes = unique(ytrue);
    prec = zeros(length(classes),1);
    for i = 1:length(classes)
        c = classes(i);
        TP = sum((ypred==c) & (ytrue==c));
        FP = sum((ypred==c) & (ytrue~=c));
        prec(i) = TP / max(TP+FP,1);
    end
    p = mean(prec);
end

function r = recall_macro(ytrue, ypred)
    classes = unique(ytrue);
    rec = zeros(length(classes),1);
    for i = 1:length(classes)
        c = classes(i);
        TP = sum((ypred==c) & (ytrue==c));
        FN = sum((ypred~=c) & (ytrue==c));
        rec(i) = TP / max(TP+FN,1);
    end
    r = mean(rec);
end

function f = f1_macro(ytrue, ypred)
    p = precision_macro(ytrue,ypred);
    r = recall_macro(ytrue,ypred);
    f = 2*p*r / max(p+r,1e-12);
end

