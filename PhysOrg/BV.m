function [mean_percent_3A,mean_percent_4A,mean_percent_5A,var_percent_3A,var_percent_4A,var_percent_5A] = BV(coordinate,Atom_label)
format short
clc
%clear
%%[sx,sy,sz]=sphere(100);%生成球的基本坐标
% coordinate=...;
% [6                  0.00000000    0.00000000    0.00000000
% 6                  -2.55424174    2.32441573    -1.68380683
% 7                  -3.55459729    2.80182891    -2.02580891
% 1                  3.38562414    1.81399164    -1.35738610
% 1                  2.13164641    0.09241345    -0.08492054
% 1                  3.39002526    3.79993327    -2.81362814
% 1                  -0.01120744    -1.02044283    -0.32166886
% 1                  1.24697536    4.82052513    -3.54472325
% 1                  -0.89886110    3.85442012    -2.81903428
% 6                  1.23278089    1.67987556    -1.24170374
% 6                  2.45032790    2.26033709    -1.67713573
% 6                  1.19534762    0.53738107    -0.40366172
% 6                  2.45192774    3.36623223    -2.48805742
% 6                  -0.00000001    2.26695120    -1.66225102
% 6                  1.23277596    3.94682928    -2.90396400
% 6                  -1.21755975    1.68649293    -1.22682131
% 6                  0.03743246    3.40944200    -2.50028907
% 6                  -1.21915077    0.58059902    -0.41590680
% 1                  -2.14776427    0.15550885    -0.09672769];%设置原子种类和坐标

%% 旋转分子坐标
AN=size(coordinate,1);%获得分子总原子数
x=coordinate(1:1:AN,2);%获得分子中所有原子的x坐标
y=coordinate(1:1:AN,3);%获得分子中所有原子的y坐标
z=coordinate(1:1:AN,4);%获得分子中所有原子的z坐标
%进行分子坐标平移
    move_x=x(Atom_label);
    move_y=y(Atom_label);
    move_z=z(Atom_label);
    x=x-move_x;
    y=y-move_y;
    z=z-move_z;
    xyz_new=[x,y,z];
%进行分子沿z轴旋转
%     cent_x=mean(x);%找到分子x坐标的几何中心
%     cent_y=mean(y);%找到分子y坐标的几何中心
%cent_z=mean(z);%找到分子z坐标的几何中心
%     angle_z=-asind(abs(cent_x)/sqrt(cent_x^2+cent_y^2));%找到沿z轴旋转的角度
%     rot_z=...
%         [cosd(angle_z),sind(angle_z),0;-sind(angle_z),cosd(angle_z),0;0,0,1];%生成沿z轴的旋转矩阵
%     xyz_new=[x,y,z]*(rot_z);%生成新的分子坐标
%     x=xyz_new(1:1:AN,1);%更新分子的x坐标
%     y=xyz_new(1:1:AN,2);%更新分子的y坐标
%     z=xyz_new(1:1:AN,3);%更新分子的z坐标
%进行分子沿x轴旋转
%     %cent_x=mean(x);%找到分子新的x坐标的几何中心
%     cent_y=mean(y);%找到分子新的y坐标的几何中心
%     cent_z=mean(z);%找到分子新的z坐标的几何中心
%     angle_x=-asind(abs(cent_y)/sqrt(cent_y^2+cent_z^2));%找到沿y轴旋转的角度
%     rot_x=...
%         [1,0,0;0,cosd(angle_x),sind(angle_x);0,-sind(angle_x),cosd(angle_x)];%生成沿z轴的旋转矩阵
%     xyz_new=[x,y,z]*rot_x;%生成新的分子坐标

%% 初始化分子坐标，测试点数及测试次数
%最终的有方向性的分子坐标
x=xyz_new(1:1:AN,1);%最终分子的x坐标
y=xyz_new(1:1:AN,2);%最终分子的y坐标
z=xyz_new(1:1:AN,3);%最终分子的z坐标
%cent_x=mean(x);%分子最终的x坐标的几何中心
%cent_y=mean(y);%分子最终的y坐标的几何中心
%cent_z=mean(z);%分子最终的z坐标的几何中心
atom=coordinate(1:1:AN,1);%获得分子中所有原子的种类
atom_rad=atom;%生成分子中所有原子的范德华半径的初始矩阵
for i=1:1:AN
    if atom(i)==1
        atom_rad(i)=1.2;
    elseif atom(i)==2
        atom_rad(i)=1.4;
    elseif atom(i)==3
        atom_rad(i)=1.82;
    elseif atom(i)==4
        atom_rad(i)=1.53;
    elseif atom(i)==5
        atom_rad(i)=1.92;
    elseif atom(i)==6
        atom_rad(i)=1.7;
    elseif atom(i)==7
        atom_rad(i)=1.55;
    elseif atom(i)==8
        atom_rad(i)=1.52;
    elseif atom(i)==9
        atom_rad(i)=1.47;
    elseif atom(i)==10
        atom_rad(i)=1.54;
    elseif atom(i)==11
        atom_rad(i)=2.27;
    elseif atom(i)==12
        atom_rad(i)=1.73;
    elseif atom(i)==13
        atom_rad(i)=1.84;
    elseif atom(i)==14
        atom_rad(i)=2.1;
    elseif atom(i)==15
        atom_rad(i)=1.8;
    elseif atom(i)==16
        atom_rad(i)=1.8;
    elseif atom(i)==17
        atom_rad(i)=1.75;
    elseif atom(i)==18
        atom_rad(i)=1.88;
    elseif atom(i)==19
        atom_rad(i)=2.75;
    elseif atom(i)==20
        atom_rad(i)=2.31;
    elseif atom(i)==35
        atom_rad(i)=1.85;
    end
end%获得分子中所有原子的范德华半径
%SSS=cell(AN,5);
%for n=1:1:AN
    %SSS{n,1}=n;
    %SSS{n,2}=atom(n);
    %SSS{n,3}= [sx.*atom_rad(n)+x(n)];
    %SSS{n,4}= [sy.*atom_rad(n)+y(n)];
    %SSS{n,5}= [sz.*atom_rad(n)+z(n)];
%end%获得分子中所有原子的原子边界坐标范围
pointnumber=500000;%定义总测试点数目
n_test = 10;%test一共进行10次

%% 测试球体的半径=3A
test_radius_3=3;%定义测试球体的半径
    percent_list_3A = zeros(1,n_test);
    for j=1:1:n_test
        percent_list_3A(j) = Test(test_radius_3,pointnumber,AN,x,y,z,atom_rad);
    end
    mean_percent_3A = mean(percent_list_3A);%将10次test得到的百分比数据进行平均并输出作为描述符
    var_percent_3A = var(percent_list_3A);%将10次test得到的方差输出

%% 测试球体的半径=4A    
test_radius_4=4;%定义测试球体的半径
    percent_list_4A = zeros(1,n_test);
    for j=1:1:n_test
        percent_list_4A(j) = Test(test_radius_4,pointnumber,AN,x,y,z,atom_rad);
    end
    mean_percent_4A = mean(percent_list_4A);%将10次test得到的百分比数据进行平均并输出作为描述符
    var_percent_4A = var(percent_list_4A);%将10次test得到的方差输出

%% 测试球体的半径=5A    
test_radius_5=5;%定义测试球体的半径
    percent_list_5A = zeros(1,n_test);
    for j=1:1:n_test
        percent_list_5A(j) = Test(test_radius_5,pointnumber,AN,x,y,z,atom_rad);
    end
    mean_percent_5A = mean(percent_list_5A);%将10次test得到的百分比数据进行平均并输出作为描述符
    var_percent_5A = var(percent_list_5A);%将10次test得到的方差输出

%% 子函数定义，采用蒙特卡洛方法实现
    function percent = Test(test_radius,pointnumber,AN,x,y,z,atom_rad)
        phi = 2*pi.*rand(pointnumber,1);
        costheta = -1 + 2.*rand(pointnumber,1);
        u = 1.*rand(pointnumber,1);

        theta = acos( costheta );
        r = test_radius .* u.^(1/3);
    
        test_x = x(1) - + r .* sin( theta) .* cos( phi );
        test_y = r .* sin( theta) .* sin( phi );
        test_z = r .* cos( theta );
        
        count = zeros(pointnumber,1);
        for n=1:1:pointnumber
            for t=1:1:AN
                if (test_x(n)-x(t))^2+(test_y(n)-y(t))^2+(test_z(n)-z(t))^2<=atom_rad(t)^2%判断点是否在原子半径范围内
                    count(n) = count(n) | 1;
                    break;
                end
            end
        end
        percent=sum(count)/pointnumber;
    end


end



