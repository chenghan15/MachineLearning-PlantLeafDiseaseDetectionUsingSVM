function [itrfin] = multisvm( TM,TL,test )
    %MULTISVM(3.0) classifies the class of given training vector according to the 
    % given group and gives us result that which class it belongs.
    % We have also to input the testing matrix

    %Inputs: TM=Training Matrix, TL=Group, test=Testing matrix
    %Outputs: itrfin=Resultant class(Group,USE ROW VECTOR MATRIX) to which testRowN set belongs 

    %----------------------------------------------------------------------%
    % IMPORTANT: DON'TM USE THIS PROGRAM FOR CLASS LESS THAN 3,            %
    %            OTHERWISE USE svmtrain,svmclassify DIRECTLY or            %
    %            add an else condition also for that case in this program. %
    %            Modify required data to use Kernel Functions and Plot also%
    %----------------------------------------------------------------------%
    %                       Date:11-08-2011(DD-MM-YYYY)                    %
    % This function for multiclass Support Vector Machine is written by
    % ANAND MISHRA (Machine Vision Lab. CEERI, Pilani, India) 
    % and this is free to use. email: anand.mishra2k88@gmail.com

    % Updated version 2.0 Date:14-10-2011(DD-MM-YYYY)
    % Updated version 3.0 Date:04-04-2012(DD-MM-YYYY)

    testSize=size(test,1);    %testMatrixSize, size(X,1),返回矩阵X的行数；
    itrfin=[];
    traininglabelGroup=TL;   %Group
    trainingMatrix=TM;   %Training Matrix
    for tempindex=1:testSize
        testRowN=test(tempindex,:);	%test tempindex row
        TL=traininglabelGroup;
        TM=trainingMatrix;
        uniqueLabel=unique(TL);   %unique(label) 返回与 label 中相同的数据，但是不包含重复项。
        uLabelLength=length(uniqueLabel);    %数组长度，即行数和列数中的较大值，相当于max(size(a)), 得到一共有几个类别(class)
        cTempTL=[];
        cTempTM=[];
        j=1;
        k=1;
        if(uLabelLength>2)     %如果类别大于2
            itr=1;
            classes=0;
            cond=max(TL)-min(TL);   %返回一个数组各不同维中的最大元素。 label最大值-最小值
            while((classes~=1) && (itr<=length(uniqueLabel)) && size(TL,2)>1 && cond>0)  %matlab里~=是不等于的意思。  size(TL,2),返回矩阵TL的列数
                %This while loop is the multiclass SVM Trick
                c1=(TL==uniqueLabel(itr));  %A = [1+i 3 2 4+i]; B = [1 3+i 2 4+i]; A == B; ans = 1x4 logical array: 0   0   1   1. 此处把label映射为两类0, 1
                                            %小括号，用于引用数组的元素。 如 X(3)就是X的第三个元素。 X([1 2 3])就是X的头三个元素。
                newClass0vs1=c1;    %class与itr 不同的class 全部映射为0, itr 的 class label 映射为1
                svmStruct = svmtrain(TM,newClass0vs1,'kernel_function','rbf');   %SVMStruct = svmtrain(Training,Group,Name,Value) returns a structure with additional options specified by one or more Name,Value pair arguments.
                classes = svmclassify(svmStruct,testRowN);     %svmclassify(SVMStruct,Sample) classifies each row of the data in Sample, a matrix of data, using the information in a support vector machine classifier structure SVMStruct, created using the svmtrain function. Like the training data used to create SVMStruct, Sample is a matrix where each row corresponds to an observation or replicate, and each column corresponds to a feature or variable. Therefore, Sample must have the same number of columns as the training data. This is because the number of columns defines the number of features. Group indicates the group to which each row of Sample has been assigned.

                %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                % This is the loop for Reduction of Training Set
                for i=1:size(newClass0vs1,2)    %size(newClass0vs1,2),返回矩阵newClass0vs1的列数
                    if newClass0vs1(1,i)==0;    %A(2,1) = {[1 2 3; 4 5 6]}，就是一个2行一列的单元数组
                        cTempTM(k,:)=TM(i,:);    %一行元素：  A(i,:)表示提取A矩阵的第i行元素，c3存储了newClass0vs1中非1的所有TM数据
                        k=k+1;
                    end
                end                    
                TM=cTempTM;  %TM赋值为c3
                cTempTM=[];
                k=1;
                
                %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
                
                %>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        

                % This is the loop for reduction of group
                for i=1:size(newClass0vs1,2)
                    if newClass0vs1(1,i)==0;
                        cTempTL(1,j)=TL(1,i);
                        j=j+1;
                    end
                end
                TL=cTempTL;  %TM赋值为c4
                cTempTL=[];
                j=1;
                
                %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<        

                %重新计算变量
                cond=max(TL)-min(TL); % Condition for avoiding group 
                                    %to contain similar type of values 
                                    %and the reduce them to process
                % This condition can select the particular value of iteration
                % base on classes
                if classes~=1
                    itr=itr+1;
                end    
            end
        end

        valt=traininglabelGroup==uniqueLabel(itr); % This logic is used to allow classification
        val=traininglabelGroup(valt==1); % of multiple rows testing matrix
        val=unique(val);
        itrfin(tempindex,:)=val;  
    end

end

% Give more suggestions for improving the program.
% From R2018a, matlab removed svmtrain and svmclassify














