clear; clc; close all;

% --- 1. VERİ YÜKLEME ---
datasetPath = fullfile('data', 'processed');

if ~exist(datasetPath, 'dir')
    error('HATA: data/processed klasörü yok!');
end

disp('Veriler yükleniyor (Bu işlem biraz sürebilir)...');
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Sınıf Özetini Göster
tbl = countEachLabel(imds);
disp('--- TESPİT EDİLEN MODELLER ---');
disp(tbl);

numClasses = height(tbl);
fprintf('\nToplam %d farklı araç modeli tespit edildi.\n', numClasses);

if numClasses < 2
    error('Eğitim için en az 2 model gerekli. Klasörlerini kontrol et!');
end

% --- 2. VERİ AYIRMA (%70 Eğitim, %30 Test) ---
% Randomized split ile veriyi karıştırıyoruz
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

% --- 3. AĞ HAZIRLIĞI (ALEXNET) ---
try
    net = alexnet;
catch
    error('AlexNet yüklü değil! Add-Ons kısmından "Deep Learning Toolbox Model for AlexNet Network" paketini yükle.');
end

inputSize = net.Layers(1).InputSize;

% RAM Tasarrufu için Resim İşleme
% Resimleri hafızaya tek seferde almaz, sırası gelince okur.
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

% Transfer Learning Katman Değişimi
layersTransfer = net.Layers(1:end-3);

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

% --- 4. GÜVENLİ EĞİTİM AYARLARI (CRITICAL) ---
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 16, ...
    'InitialLearnRate', 1e-4, ... 
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 20, ... % Her 20 adımda bir test et
    'Verbose', true, ...          % İlerlemeyi komut satırına yaz
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto'); % Varsa GPU, yoksa CPU kullanır

% --- 5. BAŞLAT ---
disp('Eğitim başlıyor... Lütfen bilgisayara dokunma, fanlar çalışabilir :)');
try
    [trainedNet, trainInfo] = trainNetwork(augimdsTrain, layers, options);
catch ME
    disp('HATA ALINDI! Muhtemelen RAM yetmedi.');
    disp('ÇÖZÜM: Kodun içindeki "MiniBatchSize" değerini 16 dan 8 e düşür.');
    rethrow(ME);
end

% --- 6. SONUÇ RAPORU ---
disp('Eğitim bitti. Sonuçlar hesaplanıyor...');
[YPred, scores] = classify(trainedNet, augimdsValidation);
YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation);
fprintf('\n>>> GENEL BAŞARI ORANI: %.2f%% <<<\n', accuracy * 100);

% Confusion Matrix
figure('Name', 'Proje Sonuç Matrisi', 'NumberTitle', 'off');
cm = confusionchart(YValidation, YPred);
cm.Title = ['Araç Tanıma Başarısı (Doğruluk: %' num2str(accuracy*100, '%.1f') ')'];

% Modeli Kaydet
save('Final_Model_20Arac.mat', 'trainedNet');
disp('Model başarıyla kaydedildi: Final_Model_20Arac.mat');