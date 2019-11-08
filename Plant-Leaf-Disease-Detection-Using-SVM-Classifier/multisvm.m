function [results] = multisvm(TM, TL, test)
    %Inputs: TM=Training Matrix, TL=Group, test=Testing matrix
    %Outputs: results,  ROW VECTOR MATRIX  which test set belongs to

    testSize = size(test, 1);    %testMatrixSize, size(X,1),返回矩阵X的行数；
    results = [];
    traininglabelGroup = TL;   %Group
    trainingMatrix = TM;   %Training Matrix
    for tempindex = 1:testSize
        testRowN = test(tempindex, :);	%test tempindex row
        TL = traininglabelGroup;
        TM = trainingMatrix;
        uniqueLabel = unique(TL);   %unique(label) 返回与 label 中相同的数据，但是不包含重复项。
        uLabelLength = length(uniqueLabel);    %数组长度，即行数和列数中的较大值，相当于max(size(a)), 得到一共有几个类别(class)

        if(uLabelLength > 2)     %如果类别大于2
            indexUniqueLabel = 1;
            classes = 0;
            cond = max(TL)-min(TL);   %返回一个数组各不同维中的最大元素。 label最大值-最小值
            while((classes ~= 1) && (indexUniqueLabel <= length(uniqueLabel)) && (size(TL, 2) > 1)  && (cond > 0))  %matlab里~=是不等于的意思。  size(TL,2),返回矩阵TL的列数
                %This while loop is the multiclass SVM Trick
                                                                            %A = [1+i 3 2 4+i]; B = [1 3+i 2 4+i]; A == B; ans = 1x4 logical array: 0   0   1   1. 此处把label映射为两类0, 1
                                                                            %小括号，用于引用数组的元素。 如 X(3)就是X的第三个元素。 X([1 2 3])就是X的头三个元素。
                newClass0vs1 = (TL == uniqueLabel(indexUniqueLabel));       %class与itr 不同的class 全部映射为0, indexUniqueLabel 的 class label 映射为1
                svmStruct = svmtrain(TM, newClass0vs1, 'kernel_function', 'rbf');   %SVMStruct = svmtrain(Training,Group,Name,Value) returns a structure with additional options specified by one or more Name,Value pair arguments.
                classes = svmclassify(svmStruct, testRowN);     %svmclassify(SVMStruct,Sample) classifies each row of the data in Sample, a matrix of data, using the information in a support vector machine classifier structure SVMStruct, created using the svmtrain function. Like the training data used to create SVMStruct, Sample is a matrix where each row corresponds to an observation or replicate, and each column corresponds to a feature or variable. Therefore, Sample must have the same number of columns as the training data. This is because the number of columns defines the number of features. Group indicates the group to which each row of Sample has been assigned.

                %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                % This is the loop for Reduction of Training Set
                tempIndex = 1;
                cTempTM = [];
                cTempTL = []; 
                for i = 1:size(newClass0vs1, 2)         %size(newClass0vs1,2),返回矩阵newClass0vs1的列数
                    if(newClass0vs1(1, i) == 0);        %A(2,1) = {[1 2 3; 4 5 6]}，就是一个2行一列的单元数组
                                                        %一行元素：  A(i,:)表示提取A矩阵的第i行元素，c3存储了newClass0vs1中非1的所有TM数据
                        cTempTM(tempIndex, :) = TM(i, :);   % Reduction of Training Set
                        cTempTL(1, tempIndex) = TL(1, i);   % reduction of group
                        tempIndex = tempIndex + 1;
                    end
                end                    
                TM = cTempTM;  %TM赋值为c3                                
                TL = cTempTL;  %TM赋值为c4
                
                %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<        

                %重新计算变量
                cond = max(TL)-min(TL); % Condition for avoiding group 
                                        %to contain similar type of values 
                                        %and the reduce them to process
                % This condition can select the particular value of iteration
                % base on classes
                if(classes ~= 1)
                    indexUniqueLabel = indexUniqueLabel + 1;
                end                                    
            end
        end

        %remapping to classes
        valt = (traininglabelGroup == uniqueLabel(indexUniqueLabel));       % This logic is used to allow classification
        val = traininglabelGroup(valt == 1);                                % of multiple rows testing matrix
        val = unique(val);
        results(tempindex, :) = val;  
    end
end

% From R2018a, matlab removed svmtrain and svmclassify














