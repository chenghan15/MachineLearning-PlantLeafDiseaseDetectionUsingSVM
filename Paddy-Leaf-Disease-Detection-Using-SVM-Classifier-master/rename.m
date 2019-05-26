
clc
clear 
file = dir('*.JPG')
len = length(file);
for i = 1 : len
    oldname = file(i).name
    newname = strcat( num2str(i+len),'.JPG');
    % '!'的意思是调用系统函数，32是ASCII码值，表示空格
    eval(['!rename' 32 oldname 32 newname]);
end 
