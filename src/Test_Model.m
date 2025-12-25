clear; clc; close all;

% --- 1. MODELİ YÜKLE ---
% Kod 'src' klasöründe olduğu için model bir üstteki 'models' klasöründedir.
modelPath = fullfile('..', 'models', 'Final_Model_ResNet50.mat');

disp('Model dosyası aranıyor...');
if exist(modelPath, 'file')
    disp(['Model bulundu: ' modelPath]);
    load(modelPath, 'trainedNet');
else
    % Eğer dosya yoksa kullanıcıya seçtir (Hata önleyici)
    disp('Model otomatik bulunamadı! Lütfen models klasöründeki .mat dosyasını seçin.');
    [file, path] = uigetfile('*.mat', 'Model Dosyasını Seç');
    if isequal(file, 0), error('Model seçilmedi.'); end
    load(fullfile(path, file), 'trainedNet');
end

% --- 2. TEST EDİLECEK RESMİ SEÇ ---
[file, path] = uigetfile({'*.jpg;*.png;*.jpeg', 'Resim Dosyaları'}, 'Test İçin Bir Araba Resmi Seçin');
if isequal(file, 0)
    disp('Resim seçilmedi.');
    return;
end

imgPath = fullfile(path, file);
img = imread(imgPath);

% --- 3. RESMİ MODELİN İSTEDİĞİ BOYUTA GETİR ---
inputSize = trainedNet.Layers(1).InputSize(1:2); % ResNet için 224x224
imgResized = imresize(img, inputSize);

% --- 4. TAHMİN YAP ---
[label, scores] = classify(trainedNet, imgResized);
confidence = max(scores) * 100; % Güven oranı

% --- 5. SONUCU GÖSTER ---
figure('Name', 'Test Sonucu', 'NumberTitle', 'off');
imshow(img);
title({['Tahmin: ' char(label)], ...
       ['Güven: %' num2str(confidence, '%.2f')]});

msgbox(['Bu araç: ' char(label) ' (Güven: %' num2str(confidence, '%.2f') ')'], 'Sonuç');
