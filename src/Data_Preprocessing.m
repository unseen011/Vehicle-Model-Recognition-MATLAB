clear; clc;

% Ayarlar
sourceDir = 'data'; % .mat dosyalarinin oldugu yer
outputDir = fullfile('data', 'processed'); % Kirpilan resimlerin gidecegi yer

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Klasordeki tum _GT.mat dosyalarini bul
files = dir(fullfile(sourceDir, '*_GT.mat'));

if isempty(files)
    error('Hic _GT.mat dosyasi bulunamadi! Dosya isimlerinin _GT.mat ile bittiginden emin ol.');
end

disp('--- Veri Hazırlama Başlıyor ---');

for i = 1:length(files)
    fileName = files(i).name;
    filePath = fullfile(sourceDir, fileName);
    
    % Dosyayi yukle
    dataStruct = load(filePath);
    varNames = fieldnames(dataStruct);
    gTruth = dataStruct.(varNames{1}); % Degisken ismini dinamik al
    
    % Etiket tablosunu al
    labelData = gTruth.LabelData;
    numImages = height(labelData);
    
    % Araba markasini dosya adindan cikart (Orn: Dacia_Duster_GT.mat -> Dacia_Duster)
    folderName = strrep(fileName, '_GT.mat', '');
    targetFolder = fullfile(outputDir, folderName);
    
    if ~exist(targetFolder, 'dir')
        mkdir(targetFolder);
    end
    
    fprintf('%s isleniyor... (%d resim)\n', folderName, numImages);
    
    % Resimleri tek tek isle
    for j = 1:numImages
        % Resim yolunu bul
        imgSource = gTruth.DataSource.Source{j};
        
        % Eğer tam yol yoksa, data klasörü ile birleştir (göreceli yol sorunu için)
        if exist(imgSource, 'file') ~= 2
             [~, name, ext] = fileparts(imgSource);
             % Varsayım: Resimler data/ModelAdi klasöründe
             imgSource = fullfile('data', folderName, [name ext]);
        end

        try
            img = imread(imgSource);
            
            % Bounding Box bilgisini al (Varsayim: Tek etiket 'Araba' veya ModelAdi)
            % Tablonun ilk sutununu aliyoruz
            bboxes = labelData{j, 1}; 
            
            if ~isempty(bboxes)
                % Eger birden fazla kutu varsa ilki al, ya da dongu kur
                % Biz simdilik en buyuk kutuyu alalim (ana araba)
                [~, maxIdx] = max(bboxes(:,3) .* bboxes(:,4)); % Alana gore max
                box = bboxes(maxIdx, :);
                
                % Kirpma islemi
                croppedImg = imcrop(img, box);
                
                % Kirpilan resmi kaydet
                saveName = sprintf('img_%d.jpg', j);
                imwrite(croppedImg, fullfile(targetFolder, saveName));
            end
        catch ME
            warning('Resim islenirken hata: %s', imgSource);
        end
    end
end

disp('--- Veri Hazırlama Tamamlandı! ---');
disp(['Islemi biten veriler surada: ' outputDir]);