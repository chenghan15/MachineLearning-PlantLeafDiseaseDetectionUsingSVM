function [results] = multisvm(TM, TL, test)
    %Inputs: TM=Training Matrix, TL=Group, test=Testing matrix
    %Outputs: results,  ROW VECTOR MATRIX  which test set belongs to

    testSize = size(test, 1);    %testMatrixSize, size(X,1),���ؾ���X��������
    results = [];
    traininglabelGroup = TL;   %Group
    trainingMatrix = TM;   %Training Matrix
    for tempindex = 1:testSize
        testRowN = test(tempindex, :);	%test tempindex row
        TL = traininglabelGroup;
        TM = trainingMatrix;
        uniqueLabel = unique(TL);   %unique(label) ������ label ����ͬ�����ݣ����ǲ������ظ��
        uLabelLength = length(uniqueLabel);    %���鳤�ȣ��������������еĽϴ�ֵ���൱��max(size(a)), �õ�һ���м������(class)

        if(uLabelLength > 2)     %���������2
            indexUniqueLabel = 1;
            classes = 0;
            cond = max(TL)-min(TL);   %����һ���������ͬά�е����Ԫ�ء� label���ֵ-��Сֵ
            while((classes ~= 1) && (indexUniqueLabel <= length(uniqueLabel)) && (size(TL, 2) > 1)  && (cond > 0))  %matlab��~=�ǲ����ڵ���˼��  size(TL,2),���ؾ���TL������
                %This while loop is the multiclass SVM Trick
                                                                            %A = [1+i 3 2 4+i]; B = [1 3+i 2 4+i]; A == B; ans = 1x4 logical array: 0   0   1   1. �˴���labelӳ��Ϊ����0, 1
                                                                            %С���ţ��������������Ԫ�ء� �� X(3)����X�ĵ�����Ԫ�ء� X([1 2 3])����X��ͷ����Ԫ�ء�
                newClass0vs1 = (TL == uniqueLabel(indexUniqueLabel));       %class��itr ��ͬ��class ȫ��ӳ��Ϊ0, indexUniqueLabel �� class label ӳ��Ϊ1
                svmStruct = svmtrain(TM, newClass0vs1, 'kernel_function', 'rbf');   %SVMStruct = svmtrain(Training,Group,Name,Value) returns a structure with additional options specified by one or more Name,Value pair arguments.
                classes = svmclassify(svmStruct, testRowN);     %svmclassify(SVMStruct,Sample) classifies each row of the data in Sample, a matrix of data, using the information in a support vector machine classifier structure SVMStruct, created using the svmtrain function. Like the training data used to create SVMStruct, Sample is a matrix where each row corresponds to an observation or replicate, and each column corresponds to a feature or variable. Therefore, Sample must have the same number of columns as the training data. This is because the number of columns defines the number of features. Group indicates the group to which each row of Sample has been assigned.

                %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                % This is the loop for Reduction of Training Set
                tempIndex = 1;
                cTempTM = [];
                cTempTL = []; 
                for i = 1:size(newClass0vs1, 2)         %size(newClass0vs1,2),���ؾ���newClass0vs1������
                    if(newClass0vs1(1, i) == 0);        %A(2,1) = {[1 2 3; 4 5 6]}������һ��2��һ�еĵ�Ԫ����
                                                        %һ��Ԫ�أ�  A(i,:)��ʾ��ȡA����ĵ�i��Ԫ�أ�c3�洢��newClass0vs1�з�1������TM����
                        cTempTM(tempIndex, :) = TM(i, :);   % Reduction of Training Set
                        cTempTL(1, tempIndex) = TL(1, i);   % reduction of group
                        tempIndex = tempIndex + 1;
                    end
                end                    
                TM = cTempTM;  %TM��ֵΪc3                                
                TL = cTempTL;  %TM��ֵΪc4
                
                %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<        

                %���¼������
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













