clear; clc; close all;

% --- 1. VERİ YÜKLEME ---
datasetPath = fullfile('data', 'processed');
if ~exist(datasetPath, 'dir'), error('Veri klasörü yok!'); end

disp('Veriler yükleniyor (ResNet için hazırlanıyor)...');
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Veriyi Karıştır ve Ayır
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

% Sınıf Sayısını Al
numClasses = numel(categories(imdsTrain.Labels));

% --- 2. RESNET-50 MODELİNİ YÜKLE ---
try
    net = resnet50;
catch
    error('ResNet-50 yüklü değil! Add-Ons kısmından indir.');
end

% ResNet Resim Boyutu (224x224)
inputSize = net.Layers(1).InputSize;

% Katman Grafiğine Çevir (Düzenleme yapmak için)
lgraph = layerGraph(net);

% --- 3. TRANSFER LEARNING AYARLARI ---
% ResNet'in son katmanlarını bulup kendi sınıf sayımıza göre değiştiriyoruz.
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

newClassLayer = classificationLayer('Name', 'new_classoutput');

% Eski katmanları yenileriyle değiştir
lgraph = replaceLayer(lgraph, 'fc1000', newLearnableLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassLayer);

% --- 4. DATA AUGMENTATION (HAFİF) ---
% ResNet zaten güçlü, çok fazla bozmaya gerek yok, hafif dokunuş yeter.
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', [-10 10], ... 
    'RandYTranslation', [-10 10]);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

% --- 5. EĞİTİM AYARLARI ---
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ...        % ResNet ağır olduğu için 16 ideal
    'MaxEpochs', 8, ...             % Daha zeki olduğu için az turda öğrenir
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 20, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu'); % GPU KULLANIMINI ZORLA

% --- 6. BAŞLAT ---
disp('ResNet-50 Motoru Çalıştırılıyor... Kemerleri Bağla! 🚀');
[trainedNet, trainInfo] = trainNetwork(augimdsTrain, lgraph, options);

% --- 7. SONUÇ ---
[YPred, scores] = classify(trainedNet, augimdsValidation);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);

fprintf('\n>>> RESNET-50 FİNAL SONUCU: %.2f%% <<<\n', accuracy * 100);

figure;
confusionchart(YValidation, YPred);
title(['ResNet-50 Başarısı: %' num2str(accuracy*100, '%.2f')]);

save('Final_Model_ResNet50.mat', 'trainedNet');