
clc
clear 
file = dir('*.JPG')
len = length(file);
for i = 1 : len
    oldname = file(i).name
    newname = strcat( num2str(i+len),'.JPG');
    % '!'����˼�ǵ���ϵͳ������32��ASCII��ֵ����ʾ�ո�
    eval(['!rename' 32 oldname 32 newname]);
end 
