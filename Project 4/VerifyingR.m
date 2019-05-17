% veriying gamma....................................
%..................................................................
clear
clc
[n1,d1,n2,d2]=Inputsys(1);
Gs1 = tf(n1,d1);
Ts=0.1;
Gd1 = c2d(Gs1,Ts,'zoh');
[num1,den1]=tfdata(Gd1,'v');
Gs2 = tf(n2,d2);
Gd2 = c2d(Gs2,Ts,'zoh');
[num2,den2]=tfdata(Gd2,'v');
sys_info = stepinfo(Gd1);
ts1 = sys_info.SettlingTime;
tr1=sys_info.RiseTime; 
sys_info = stepinfo(Gd2);
ts2 = sys_info.SettlingTime;
tr2=sys_info.RiseTime;
t=1:Ts:80;
 [g1,t1] = step(Gd1,t);
 [g2,t2] = step(Gd2,t);
P1=floor(tr1/Ts);
P2=floor(tr2/Ts);
N1=floor( ts1/Ts);
N2=floor( ts2/Ts);
P=max(P1,P2);
N=max(N1,N2);
M=P;
n=3;
m1=2;
miu=[3 5];
ui1=randn(1,length(t));
ui2=randn(1,length(t));
ui=[ui1; ui2];
Gd=[Gd1 Gd2];
[yi,t1]=lsim(Gd,ui,t);
theta1=zeros(2*miu(1)+3*n,length(t)-100);
theta2=zeros(2*miu(2)+3*n,length(t)-100);
for i=1:2
    phi1=zeros(2*miu(i)+3*n,length(t));
    for j=1:length(t)-100
        y(1,j)=yi(j+miu(i));
        for k=1:miu(i)
            if (j+miu(i)-k-1)<=0
                du(j+miu(i)-k)=ui1(j+miu(i)-k);
                du2(j+miu(i)-k)=ui2(j+miu(i)-k);
            else
                du(j+miu(i)-k)=ui1(j+miu(i)-k)-ui1(j+miu(i)-k-1);
                du2(j+miu(i)-k)=ui2(j+miu(i)-k)-ui2(j+miu(i)-k-1);
            end
            phi1(k,j)=du(j+miu(i)-k);
            phi1(k+miu(i),j)=du2(j+miu(i)-k);
        end
        for k=1:n
            if (j-k-1)<=0
                phi1(k+2*miu(i),j)=0;
                phi1(k+2*miu(i)+n,j)=0;
            else
                phi1(k+2*miu(i),j)=ui1(j-k)-ui1(j-k-1);
                phi1(k+2*miu(i)+n,j)=ui2(j-k)-ui2(j-k-1);
            end
            if (j-k)<=0
                 phi1(k+2*miu(i)+2*n,j)=0;
            else
                 phi1(k+2*miu(i)+2*n,j)=yi(j-k);  
            end
        end  
        if i==1
           %theta1(:,j)=(phi1(:,j)*phi1(:,j)')\(phi1(:,j)*y(1,j));  % we could use this method (right inverse)
           theta1(:,j)=pinv(phi1(:,j)')*y(1,j);
        else
            %theta2  (:,j)=(phi1(:,j)*phi1(:,j)')\(phi1(:,j)*y(1,j));  % we could use this method (right inverse)
            theta2(:,j)=pinv(phi1(:,j)')*y(1,j);
        end
    end
end
%..........................................................................................................................................................
 G1i=zeros(2,2,length(t)-100); % these comments are related to the state which we use IIR for the future
 G2i=zeros(2,2,length(t)-100);
M1_=zeros(2,n,length(t)-100);
M2_=zeros(2,n,length(t)-100);
F=zeros(2,n,length(t)-100);
for q=1:length(t)-100
     G1i(:,:,q)=[theta1(miu(1),q) theta1(miu(1)-m1,q); theta2(miu(2),q) theta2(miu(2)-m1,q)];
    G2i(:,:,q)=[theta1(2*miu(1),q) theta1(2*miu(1)-m1,q); theta2(2*miu(2),q) theta2(2*miu(2)-m1,q)];
    M1_(:,:,q)=[theta1(2*miu(1)+1:n+2*miu(1),q)'; theta2(2*miu(2)+1:n+2*miu(2),q)'];
    M2_(:,:,q)=[theta1(2*miu(1)+1+n:2*n+2*miu(1),q)'; theta2(2*miu(2)+n+1:2*n+2*miu(2),q)'];
    F(:,:,q)=[theta1(2*miu(1)+2*n+1:3*n+2*miu(1),q)'; theta2(2*miu(2)+2*n+1:2*miu(2)+3*n,q)'];
end
Gi=[G1i G2i];
%.....................Toeplitz Matrix.................................
b1 = zeros(1,P); b1(1,1)= g1(2);
a1 = g1(2:P+1);
G1 = toeplitz(a1,b1);
G1(:,M) = G1(:,M:P)*ones(P-M+1,1);
G1 = G1(:,1:M);
%........................................................
b2 = zeros(1,P); b2(1,1)= g2(2);
a2 = g2(2:P+1);
G2 = toeplitz(a2,b2);
G2(:,M) = G2(:,M:P)*ones(P-M+1,1);
G2 = G2(:,1:M);
%......................................
G11=[G1(miu(1),1) G1(miu(1),m1+1); G1(miu(2),1) G1(miu(2),m1+1)];
G21=[G2(miu(1),1) G2(miu(1),m1+1); G2(miu(2),1) G2(miu(2),m1+1)];
G=[G11 G21];
M_=[M1_ M2_];
%....................................................................................................................
%......... gamma=1......................................
gamma =1;
gain_DC=(num1(1)+num1(2)+num1(3))/(den1(1)+den1(2)+den1(3));
gain_DC2=(num2(1)+num2(2)+num2(3))/(den2(1)+den2(2)+den2(3));
Q = eye(2);
R1 =((1.2)^2)*gamma*gain_DC^2*eye(2);
R2=gamma*gain_DC2^2*eye(2);
R=[R1 zeros(2); zeros(2) R2];
alpha=0.5;
%for q=1:length(t)-100
    Kpfc=(G'*Q*G+R)\(G'*Q);
%end
%.................................................................................................................

x01=0.0882;
x02=441.2;
dU1_=zeros(n,length(t)-100);
dU2_=zeros(n,length(t)-100);
dU_=[dU1_;dU2_];
d1=zeros(1,length(t)-100);
y1=441.2;
u_1=[];
u_2=[];
ym=[];
y=0;
Y_d=zeros(2,length(t)-100);
Y_past=zeros(2,length(t)-100);
Y_m=zeros(2,length(t)-100);
D=zeros(2,length(t)-100);
E=zeros(2,length(t)-100);
dU1=zeros(2,length(t)-100);
dU2=zeros(2,length(t)-100);
dU=[dU1;dU2];
U1=zeros(M,length(t)-100);
U2=zeros(M,length(t)-100);
Y_=zeros(n,length(t)-100);
%..................step...........................
r =ones(length(t)-100,1);
%.....................................................
for i=1:length(t)-101
    
for j=1:2
  Y_d(j,i+1)=(alpha^j)*y+(1-(alpha)^j)*r(i+1); % Programmed
end 

Y_past(:,i+1)=M_(:,:,i+1)*dU_(:,i+1)+F(:,:,i+1)*Y_(:,i+1);
D(:,i+1)=d1(i+1)*ones(2,1);

E(:,i+1)=Y_d(:,i+1)-Y_past(:,i+1)-D(:,i+1);

dU(:,i+1)=Kpfc*E(:,i+1);
dU1(:,i+1)=dU(1:2,i+1);
dU2(:,i+1)=dU(3:4,i+1);
U1(1,i+1)=dU1(1,i+1)+U1(1,i);
U2(1,i+1)=dU2(1,i+1)+U2(1,i);
dU(:,i+1)=[dU1(:,i+1);dU2(:,i+1)];

Y_m(:,i+1)=G*dU(:,i+1)+Y_past(:,i+1);

dU1_(2:n,i+2) = dU1_(1:n-1,i+1);
dU1_(1,i+2)=dU1(1,i+1);
dU2_(2:n,i+2) = dU2_(1:n-1,i+1);
dU2_(1,i+2)=dU2(1,i+1);
dU_(:,i+2)=[dU1_(:,i+2);dU2_(:,i+2)];
%Y_(2:n,i+2+miu(1))=Y_(1:n-1,i+1+miu(1)); % without using y(t+1) we can make Y_past in this way
% Y_(1,i+2+miu(1))=Y_m(1,i+1);

u1=U1(1,i+1);
u2=U2(1,i+1);
sim('Model');
d1(i+2)=y(end)-yl(end);
Y_(2:n,i+2)=Y_(1:n-1,i+1);
Y_(1,i+2)=yl(end);
y=y(end);%+dist(i,1);    % nonlinear
y1=[y1; y+441.2];
ym=[ym; yl(end)];
u_1=[u_1; u1];
u_2=[u_2; u2];
%noise=[noise; n];
x01=x1(end);
x02=x2(end);

end
figure(3);
subplot(2,2,1:2);
plot(y1,'b');
grid on
title('Response of the nonlinear system');
xlabel('sample');
subplot(2,2,3);
plot(u_1,'b');
grid on
xlabel('sample');
title('Control law for input 1 without bias');
subplot(2,2,4);
plot(u_2,'b');
grid on
xlabel('sample');
title('Control law for input 2 without bias');
%...........................................................................................
%......... gamma=1/60......................................
gamma =1/60;
gain_DC=(num1(1)+num1(2)+num1(3))/(den1(1)+den1(2)+den1(3));
gain_DC2=(num2(1)+num2(2)+num2(3))/(den2(1)+den2(2)+den2(3));
Q = eye(2);
R1 =((1.2)^2)*gamma*gain_DC^2*eye(2);
R2=gamma*gain_DC2^2*eye(2);
R=[R1 zeros(2); zeros(2) R2];
alpha=0.5;
%for q=1:length(t)-100
    Kpfc=(G'*Q*G+R)\(G'*Q);
%end
%.................................................................................................................
n=3;
x01=0.0882;
x02=441.2;
dU1_=zeros(n,length(t)-100);
dU2_=zeros(n,length(t)-100);
dU_=[dU1_;dU2_];
d1=zeros(1,length(t)-100);
y1=441.2;
u_1=[];
u_2=[];
ym=[];
y=0;
Y_d=zeros(2,length(t)-100);
Y_past=zeros(2,length(t)-100);
Y_m=zeros(2,length(t)-100);
D=zeros(2,length(t)-100);
E=zeros(2,length(t)-100);
dU1=zeros(2,length(t)-100);
dU2=zeros(2,length(t)-100);
dU=[dU1;dU2];
U1=zeros(M,length(t)-100);
U2=zeros(M,length(t)-100);
Y_=zeros(n,length(t)-100);
%..................step...........................
r =ones(length(t)-100,1);
%.....................................................
for i=1:length(t)-101
    
for j=1:2
  Y_d(j,i+1)=(alpha^j)*y+(1-(alpha)^j)*r(i+1); % Programmed
end 

Y_past(:,i+1)=M_(:,:,i+1)*dU_(:,i+1)+F(:,:,i+1)*Y_(:,i+1);
D(:,i+1)=d1(i+1)*ones(2,1);

E(:,i+1)=Y_d(:,i+1)-Y_past(:,i+1)-D(:,i+1);

dU(:,i+1)=Kpfc*E(:,i+1);
dU1(:,i+1)=dU(1:2,i+1);
dU2(:,i+1)=dU(3:4,i+1);
U1(1,i+1)=dU1(1,i+1)+U1(1,i);
U2(1,i+1)=dU2(1,i+1)+U2(1,i);
dU(:,i+1)=[dU1(:,i+1);dU2(:,i+1)];

Y_m(:,i+1)=G*dU(:,i+1)+Y_past(:,i+1);

dU1_(2:n,i+2) = dU1_(1:n-1,i+1);
dU1_(1,i+2)=dU1(1,i+1);
dU2_(2:n,i+2) = dU2_(1:n-1,i+1);
dU2_(1,i+2)=dU2(1,i+1);
dU_(:,i+2)=[dU1_(:,i+2);dU2_(:,i+2)];
%Y_(2:n,i+2+miu(1))=Y_(1:n-1,i+1+miu(1)); % without using y(t+1) we can make Y_past in this way
% Y_(1,i+2+miu(1))=Y_m(1,i+1);

u1=U1(1,i+1);
u2=U2(1,i+1);
sim('Model');
d1(i+2)=y(end)-yl(end);
Y_(2:n,i+2)=Y_(1:n-1,i+1);
Y_(1,i+2)=yl(end);
y=y(end);%+dist(i,1);    % nonlinear
y1=[y1; y+441.2];
ym=[ym; yl(end)];
u_1=[u_1; u1];
u_2=[u_2; u2];
%noise=[noise; n];
x01=x1(end);
x02=x2(end);

end
figure(3);
subplot(2,2,1:2);
hold on
plot(y1,'c');
subplot(2,2,3);
hold on
plot(u_1,'c');
subplot(2,2,4);
hold on
plot(u_2,'c');
%...................................................................................
%......... gamma=0.006......................................
gamma =0.006;
gain_DC=(num1(1)+num1(2)+num1(3))/(den1(1)+den1(2)+den1(3));
gain_DC2=(num2(1)+num2(2)+num2(3))/(den2(1)+den2(2)+den2(3));
Q = eye(2);
R1 =((1.2)^2)*gamma*gain_DC^2*eye(2);
R2=gamma*gain_DC2^2*eye(2);
R=[R1 zeros(2); zeros(2) R2];
alpha=0.5;
%for q=1:length(t)-100
    Kpfc=(G'*Q*G+R)\(G'*Q);
%end
%.................................................................................................................
n=3;
x01=0.0882;
x02=441.2;
dU1_=zeros(n,length(t)-100);
dU2_=zeros(n,length(t)-100);
dU_=[dU1_;dU2_];
d1=zeros(1,length(t)-100);
y1=441.2;
u_1=[];
u_2=[];
ym=[];
y=0;
Y_d=zeros(2,length(t)-100);
Y_past=zeros(2,length(t)-100);
Y_m=zeros(2,length(t)-100);
D=zeros(2,length(t)-100);
E=zeros(2,length(t)-100);
dU1=zeros(2,length(t)-100);
dU2=zeros(2,length(t)-100);
dU=[dU1;dU2];
U1=zeros(M,length(t)-100);
U2=zeros(M,length(t)-100);
Y_=zeros(n,length(t)-100);
%..................step...........................
r =ones(length(t)-100,1);
%.....................................................
for i=1:length(t)-101
    
for j=1:2
  Y_d(j,i+1)=(alpha^j)*y+(1-(alpha)^j)*r(i+1); % Programmed
end 

Y_past(:,i+1)=M_(:,:,i+1)*dU_(:,i+1)+F(:,:,i+1)*Y_(:,i+1);
D(:,i+1)=d1(i+1)*ones(2,1);

E(:,i+1)=Y_d(:,i+1)-Y_past(:,i+1)-D(:,i+1);

dU(:,i+1)=Kpfc*E(:,i+1);
dU1(:,i+1)=dU(1:2,i+1);
dU2(:,i+1)=dU(3:4,i+1);
U1(1,i+1)=dU1(1,i+1)+U1(1,i);
U2(1,i+1)=dU2(1,i+1)+U2(1,i);
dU(:,i+1)=[dU1(:,i+1);dU2(:,i+1)];

Y_m(:,i+1)=G*dU(:,i+1)+Y_past(:,i+1);

dU1_(2:n,i+2) = dU1_(1:n-1,i+1);
dU1_(1,i+2)=dU1(1,i+1);
dU2_(2:n,i+2) = dU2_(1:n-1,i+1);
dU2_(1,i+2)=dU2(1,i+1);
dU_(:,i+2)=[dU1_(:,i+2);dU2_(:,i+2)];
%Y_(2:n,i+2+miu(1))=Y_(1:n-1,i+1+miu(1)); % without using y(t+1) we can make Y_past in this way
% Y_(1,i+2+miu(1))=Y_m(1,i+1);

u1=U1(1,i+1);
u2=U2(1,i+1);
sim('Model');
d1(i+2)=y(end)-yl(end);
Y_(2:n,i+2)=Y_(1:n-1,i+1);
Y_(1,i+2)=yl(end);
y=y(end);%+dist(i,1);    % nonlinear
y1=[y1; y+441.2];
ym=[ym; yl(end)];
u_1=[u_1; u1];
u_2=[u_2; u2];
%noise=[noise; n];
x01=x1(end);
x02=x2(end);

end
figure(3);
subplot(2,2,1:2);
hold on
plot(y1,'m');
hold on
plot(r+441.2,'r');
legend('gamma=1','gamma=1/60','gamma=0.006','r');

subplot(2,2,3);
hold on
plot(u_1,'m');
legend('gamma=1','gamma=1/60','gamma=0.006');
subplot(2,2,4);
hold on
plot(u_2,'m');
legend('gamma=1','gamma=1/60','gamma=0.006');
