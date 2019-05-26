%% Project Title: Paddy Leaf Disease Detection

clc
close all 
clear all

while (1==1)
    choice=menu('Paddy Leaf Disease Detection','....... Training........','....... Testing......','........ Close........');
    
    if (choice==1)
        %% Image Read
        xx = 1;
        for k=1:12
            I = imread(sprintf('C:/Users/az91d/Desktop/newFolder/Paddy-Leaf-Disease-Detection-Using-SVM-Classifier-master_v001/Paddy-Leaf-Disease-Detection-Using-SVM-Classifier-master/Train/Train (%d).jpg', k));
            I = imresize(I,[1024,1024]);
            [I3,RGB] = createMask(I);
            seg_img = RGB;
            img = rgb2gray(seg_img);
            glcms = graycomatrix(img);
            
            stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');

            Contrast = stats.Contrast;
            Energy = stats.Energy;
            Homogeneity = stats.Homogeneity;
            Mean = mean2(seg_img);
            Standard_Deviation = std2(seg_img);
            Entropy = entropy(seg_img);
            RMS = mean2(rms(seg_img));
            %Skewness = skewness(img)
            Variance = mean2(var(double(seg_img)));
            a = sum(double(seg_img(:)));
            Smoothness = 1-(1/(1+a));
            % Inverse Difference Movement
            m = size(seg_img,1);
            n = size(seg_img,2);
            in_diff = 0;
            for i = 1:m
                for j = 1:n
                    temp = seg_img(i,j)./(1+(i-j).^2);
                    in_diff = in_diff+temp;
                end
            end
            IDM = double(in_diff);

            ff = [Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM];
            
            if k==1
                Train_Feat = ff;
            else
                Train_Feat = [Train_Feat;ff];
            end
            
            if k<=3 && k>1
                xx = [xx;1];    % 1 is diseased                
            elseif k<=6 && k>3
                xx = [xx;2];    % 2 is not diseased
            elseif k<=9 && k>6
                xx = [xx;3];    % 2 is not diseased 
            elseif k<=12 && k>9
                xx = [xx;4];    % 2 is not diseased                 
            end
            Train_Label = xx.';
            Train_Label = transpose(xx);
        end
        disp('Train Complete');
        writeMatrix(Train_Feat, 'Train_Feat.txt');
        writeMatrix(Train_Label, 'Train_Label.txt');      
        save('Training_Data.mat','Train_Feat','Train_Label');
    end
    if (choice==2)
        
        [filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick a Leaf Image File');
        I = imread([pathname,filename]);
        I = imresize(I,[1024,1024]);
        figure, imshow(I); title('Query Leaf Image');

        
        %% Create Mask Or Segmentation Image
        [I3,RGB] = createMask(I);
        seg_img = RGB;
        figure, imshow(I3); title('BW Image');
        figure, imshow(seg_img); title('Segmented Image');
        
        
        %% Feature Extraction
        % Convert to grayscale if image is RGB
        img = rgb2gray(seg_img);
        %figure, imshow(img); title('Gray Scale Image');

        % Create the Gray Level Cooccurance Matrices (GLCMs)
        glcms = graycomatrix(img);

        % Derive Statistics from GLCM
        stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');

        Contrast = stats.Contrast;
        Energy = stats.Energy;
        Homogeneity = stats.Homogeneity;
        Mean = mean2(seg_img);
        Standard_Deviation = std2(seg_img);
        Entropy = entropy(seg_img);
        RMS = mean2(rms(seg_img));
        %Skewness = skewness(img)
        Variance = mean2(var(double(seg_img)));
        a = sum(double(seg_img(:)));
        Smoothness = 1-(1/(1+a));
        % Inverse Difference Movement
        m = size(seg_img,1);
        n = size(seg_img,2);
        in_diff = 0;
        for i = 1:m
            for j = 1:n
                temp = seg_img(i,j)./(1+(i-j).^2);
                in_diff = in_diff+temp;
            end
        end
        IDM = double(in_diff);

        feat_disease = [Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM];
        %% SVM Classifier
        % Load All The Features
        %load('Training_Data.mat')

        % Put the test features into variable 'test'
        test = feat_disease;
        writeMatrix(test, 'Test_Feat.txt');  
        save('Test_Data.mat','test','Train_Label');
        result = multisvm(Train_Feat,Train_Label,test);
        disp('Result:');
        disp(result);

        
        %% Visualize Results
        if result == 1
            helpdlg(' Disease Detect ');
            disp(' Disease Detect ');
        else
            helpdlg(' Disease not Detect ');
            disp('Disease not Detect');
        end
    end
    if (choice==3)
        close all;
        return;
    end
end