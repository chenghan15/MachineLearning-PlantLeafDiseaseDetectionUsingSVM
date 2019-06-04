%% Project Title: Paddy Leaf Disease Detection

clc
close all 
clear all

while (1==1)
    choice=menu('Paddy Leaf Disease Detection','....... Training........','....... Testing......','........ Close........');
    
    %currModel = 'single';
    currModel = 'batchs';    
    
    if (choice==1)
        %% Image Read
        xx = 1;
        index = 1;
        
        
%         for k=1:100     %504
%             I = imread(sprintf('C:/Users/az91d/Desktop/newFolder/Paddy-Leaf-Disease-Detection-Using-SVM-Classifier-master_v005/Paddy-Leaf-Disease-Detection-Using-SVM-Classifier-master/Train/Apple___Apple_scab/%d.JPG', k));
%             I = imresize(I,[256,256]);
%             [I3,RGB] = createMask(I);
%             seg_img = RGB;
%             img = rgb2gray(seg_img);
%             glcms = graycomatrix(img);
%             
%             stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
% 
%             Contrast = stats.Contrast;
%             
% 
%             
%             Energy = stats.Energy;
%             Homogeneity = stats.Homogeneity;
%             Mean = mean2(seg_img);
%             Standard_Deviation = std2(seg_img);
%             Entropy = entropy(seg_img);
%             RMS = mean2(rms(seg_img));
%             %Skewness = skewness(img)
%             Variance = mean2(var(double(seg_img)));
%             a = sum(double(seg_img(:)));
%             Smoothness = 1-(1/(1+a));
%             
%             Correlation = stats.Correlation;            
%             Kurtosis = kurtosis(double(seg_img(:)));
%             Skewness = skewness(double(seg_img(:)));            
%             
%             % Inverse Difference Movement
%             m = size(seg_img,1);
%             n = size(seg_img,2);
%             in_diff = 0;
%             for i = 1:m
%                 for j = 1:n
%                     temp = seg_img(i,j)./(1+(i-j).^2);
%                     in_diff = in_diff+temp;
%                 end
%             end
%             IDM = double(in_diff);
% 
%             ff = [Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM, Correlation, Kurtosis, Skewness];
%             
% 
%             
%             if index==1
%                 Train_Feat = ff;
%             else
%                 Train_Feat = [Train_Feat;ff];
%             end
%             
%           
%             
%             
%             disp('Current: Apple___Apple_scab ');
%             disp(k);
% 
%             if(index > 1)
%                 xx = [xx;4];    % 1 is Apple___Apple_scab                
%             end
%             
%             index = index+1;  
%              
%             Train_Label = xx.';
%             Train_Label = transpose(xx);
%         end
        
        for k=1:200     %497
            I = imread(sprintf('C:/Users/az91d/Desktop/newFolder/Paddy-Leaf-Disease-Detection-Using-SVM-Classifier-master_v005/Paddy-Leaf-Disease-Detection-Using-SVM-Classifier-master/Train/Apple___Black_rot/%d.jpg', k));
            I = imresize(I,[256,256]);
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
            
            Correlation = stats.Correlation;            
            Kurtosis = kurtosis(double(seg_img(:)));
            Skewness = skewness(double(seg_img(:)));             
            
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

            ff = [Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM, Correlation, Kurtosis, Skewness];
            

            
            if index==1
                Train_Feat = ff;
            else
                Train_Feat = [Train_Feat;ff];
            end
            
          
            
            disp('Current: Apple___Black_rot ');
            disp(k);            
            
            if(index > 1)
                xx = [xx;1];    % 2 is Apple___Black_rot
            end
            
            index = index+1;  
            
            Train_Label = xx.';
            Train_Label = transpose(xx);
        end        
        
        for k=1:200     %220
            I = imread(sprintf('C:/Users/az91d/Desktop/newFolder/Paddy-Leaf-Disease-Detection-Using-SVM-Classifier-master_v005/Paddy-Leaf-Disease-Detection-Using-SVM-Classifier-master/Train/Apple___Cedar_apple_rust/%d.jpg', k));
            I = imresize(I,[256,256]);
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
            
            Correlation = stats.Correlation;            
            Kurtosis = kurtosis(double(seg_img(:)));
            Skewness = skewness(double(seg_img(:)));             
            
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

            ff = [Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM, Correlation, Kurtosis, Skewness];
            

            
            if index==1
                Train_Feat = ff;
            else
                Train_Feat = [Train_Feat;ff];
            end
            
          
            
            disp('Current: Apple___Cedar_apple_rust ');
            disp(k); 
            
            if(index > 1)
                xx = [xx;2];    % 2 is Apple___Cedar_apple_rust
            end
            
            index = index+1;              

            Train_Label = xx.';
            Train_Label = transpose(xx);
        end          
        
        for k=1:200         %1316
            I = imread(sprintf('C:/Users/az91d/Desktop/newFolder/Paddy-Leaf-Disease-Detection-Using-SVM-Classifier-master_v005/Paddy-Leaf-Disease-Detection-Using-SVM-Classifier-master/Train/Apple___healthy/%d.jpg', k));
            I = imresize(I,[256,256]);
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
            
            Correlation = stats.Correlation;            
            Kurtosis = kurtosis(double(seg_img(:)));
            Skewness = skewness(double(seg_img(:)));             
            
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

            ff = [Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM, Correlation, Kurtosis, Skewness];
            

            
            if index==1
                Train_Feat = ff;
            else
                Train_Feat = [Train_Feat;ff];
            end
            
         
            
            disp('Current: Apple___healthy ');
            disp(k); 
            
            if(index > 1)
                xx = [xx;3];    % 2 is Apple___healthy       
            end
            
            index = index+1;              

            Train_Label = xx.';
            Train_Label = transpose(xx);
        end          
                
        disp('Train Complete');
        writeMatrix(Train_Feat, 'Train_Feat.txt');
        writeMatrix(Train_Label, 'Train_Label.txt');      
        save('Training_Data.mat','Train_Feat','Train_Label');
        

    end
    if (choice==2)
        
        load('Training_Data.mat');
        
        if('single' == currModel)
            disp('single model');
            
            [filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick a Leaf Image File');
            I = imread([pathname,filename]);
            I = imresize(I,[256,256]);
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

            Correlation = stats.Correlation;            
            Kurtosis = kurtosis(double(seg_img(:)));
            Skewness = skewness(double(seg_img(:)));         

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

            feat_disease = [Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM, Correlation, Kurtosis, Skewness];     
            
            
            Train_Feat = [Train_Feat;feat_disease];


            Train_Label = Train_Label'; 
            Train_Label = [Train_Label;1].';             
        elseif('batchs' == currModel)
            disp('batch model');
            
            for k=1:55         %1316
                I = imread(sprintf('C:/Users/az91d/Desktop/newFolder/Paddy-Leaf-Disease-Detection-Using-SVM-Classifier-master_v005/Paddy-Leaf-Disease-Detection-Using-SVM-Classifier-master/Test/Apple___Cedar_apple_rust/%d.jpg', k));
                I = imresize(I,[256,256]);
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

                Correlation = stats.Correlation;            
                Kurtosis = kurtosis(double(seg_img(:)));
                Skewness = skewness(double(seg_img(:)));             

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

                ff = [Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM, Correlation, Kurtosis, Skewness];




                disp('Current: Apple___test ');
                disp(k); 



                Train_Feat = [Train_Feat;ff];
                Train_Label = Train_Label'; 
                Train_Label = [Train_Label;4].';              

            end                  
        end;
            
        
        
        
        
%         test = feat_disease;
%         writeMatrix(test, 'Test_Feat.txt');  
%         save('Test_Data.mat','test','Train_Label');
        
        instance = sparse(Train_Feat);
        libsvmwrite('Train_Feat_svm_data.mat',Train_Label', instance); 
        [s,e]=dos('svm-scale.bat');
        [data_label,data_instance]=libsvmread('scaled_Train_Feat_svm_data.mat');  
        model = svmtrain(data_label(1:600),data_instance(1:600,:), '-b 1 -t 2');        % -t 2  use rbf kernel
        
        if('single' == currModel)
            [predict_label,accuracy,dec_values] = svmpredict(data_label(601:601),data_instance(601:601,:), model, '-b 1'); 
        elseif('batchs' == currModel)
            [predict_label,accuracy,dec_values] = svmpredict(data_label(601:655),data_instance(601:655,:), model, '-b 1');    
        end
          
        
        
        disp(predict_label);
        disp(accuracy);          
        disp(dec_values);   
        
        uniqueLabel = unique(data_label);
        
        [~,mdex]=max(dec_values(1,:));
        p=uniqueLabel(mdex);
        
        writeMatrix(predict_label, 'TestResult.txt');
        writeMatrix(dec_values, 'TestPredictValue.txt');        
        
        result = predict_label;
        
        
        
%         instance = sparse(test);
%         tesl = [1];
%         libsvmwrite('libsvm_test_data.mat', tesl', instance); 
%         [s,e]=dos('svm-scale_test.bat');
%         [data_label,data_instance]=libsvmread('scaled_libsvm_data.mat');  
%         model = svmtrain(data_label,data_instance)  
%         [test_data_label,test_data_instance]=libsvmread('scaled_libsvm_test_data.mat'); 
%         [predict_label,accuracy,dec_values] = svmpredict(tesl, instance, model, '-b 1')         
        
        
        
        
        %result = multisvm(Train_Feat,Train_Label,test);
        %result = 'unknow~~~';
        disp('Result:');
        disp(result);

        
        %% Visualize Results
        if result == 4
            helpdlg('Apple___Apple_scab Disease Detect');
            disp('Apple___Apple_scab Disease Detect');
        elseif result == 1
            helpdlg('Apple___Black_rot Disease Detect');
            disp('Apple___Black_rot Disease Detect');
        elseif result == 2
            helpdlg('Apple___Cedar_apple_rust Disease Detect');
            disp('Apple___Cedar_apple_rust Disease Detect');  
        elseif result == 3
            helpdlg(' Disease not Detect ');
            disp('Disease not Detect');            
        end
    end
    if (choice==3)
        close all;
        return;
    end
end