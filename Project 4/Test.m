% clear
% clc
% [n1,d1,n2,d2]=Inputsys(1);
% Gs1 = tf(n1,d1);
% Ts=0.1;
% Gd1 = c2d(Gs1,Ts,'zoh');
% [num1,den1]=tfdata(Gd1,'v');
% Gs2 = tf(n2,d2);
% Gd2 = c2d(Gs2,Ts,'zoh');
% [num2,den2]=tfdata(Gd2,'v');
% sys_info = stepinfo(Gd1);
% ts1 = sys_info.SettlingTime;
% tr1=sys_info.RiseTime; 
% sys_info = stepinfo(Gd2);
% ts2 = sys_info.SettlingTime;
% tr2=sys_info.RiseTime;
% t=1:Ts:30;
% [g1,t1] = step(Gd1,t);
% [g2,t2] = step(Gd2,t);
% P1=floor(tr1/Ts);
% P2=floor(tr2/Ts);
% N1=floor( ts1/Ts);
% N2=floor( ts2/Ts);
% P=max(P1,P2);
% %P=30;
% N=max(N1,N2);
% M=P;
% n=10;
% m1=2;
% miu=[3 5];
% % theta11=zeros(miu(1)+2*n,length(t)-100);
% % theta21=zeros(miu(2)+2*n,length(t)-100);
% % for i=1:2
% %     ui=randn(1,length(t));
% %     [yi,t1]=lsim(Gd1,ui,t);
% %     phi1=zeros(miu(i)+2*n,length(t));
% %     for j=1:length(t)-100
% %         y(1,j)=yi(j+miu(i));
% %         for k=1:miu(i)
% %             if (j+miu(i)-k-1)<=0
% %                 du(j+miu(i)-k)=ui(j+miu(i)-k);
% %             else
% %                 du(j+miu(i)-k)=ui(j+miu(i)-k)-ui(j+miu(i)-k-1);
% %             end
% %             phi1(k,j)=du(j+miu(i)-k);
% %         end
% %         for k=1:n
% %             if (j-k-1)<=0
% %                 phi1(k+miu(i),j)=0;
% %             else
% %                 phi1(k+miu(i),j)=ui(j-k)-ui(j-k-1);
% %             end
% %             if (j-k)<=0
% %                  phi1(k+miu(i)+n,j)=0;
% %             else
% %                  phi1(k+miu(i)+n,j)=yi(j-k);  
% %             end
% %         end  
% %         if i==1
% %             theta11(:,j)=(phi1(:,j)'*phi1(:,j))\(phi1(:,j)'*y(1,j));
% %         else
% %             theta21(:,j)=(phi1(:,j)'*phi1(:,j))\(phi1(:,j)'*y(1,j));
% %         end
% %     end
% % end
% % %..........................................................................................................................................................
% % theta12=zeros(2*n+miu(1),length(t)-100);
% % theta22=zeros(2*n+miu(2),length(t)-100);
% % for i=1:2
% %     ui=randn(1,length(t));
% %     [yi2,t1]=lsim(Gd2,ui,t);
% %     phi2=zeros(2*n+miu(i),length(t));
% %     for j=1:length(t)-100
% %         y2(1,j)=yi2(j+miu(i));
% %         for k=1:miu(i)
% %             if (j+miu(i)-k-1)<=0
% %                 du(j+miu(i)-k)=ui(j+miu(i)-k);
% %             else
% %                 du(j+miu(i)-k)=ui(j+miu(i)-k)-ui(j+miu(i)-k-1);
% %             end
% %             phi2(k,j)=du(j+miu(i)-k);
% %         end
% %         for k=1:n
% %             if (j-k-1)<=0
% %                 phi2(k+miu(i),j)=0;
% %             else
% %                 phi2(k+miu(i),j)=ui(j-k)-ui(j-k-1);
% %             end
% %             if (j-k)<=0
% %                  phi2(k+n+miu(i),j)=0;
% %             else
% %                  phi2(k+n+miu(i),j)=yi2(j-k);  
% %             end
% %         end
% %         if i==1
% %             theta12(:,j)=(phi2(:,j)'*phi2(:,j))\(phi2(:,j)'*y2(1,j));
% %         else
% %             theta22(:,j)=(phi2(:,j)'*phi2(:,j))\(phi2(:,j)'*y2(1,j));
% %         end
% %     end
% % end
% % G1=zeros(2,2);
% % G2=zeros(2,2);
% % G1=[g1(miu(1)) g1(miu(1)-m1); g1(miu(2)) g1(miu(2)-m1)];
% % G2=[g2(miu(1)) g2(miu(1)-m1); g2(miu(2)) g2(miu(2)-m1)];
% % M1_=zeros(2,n,length(t)-100);
% % M2_=zeros(2,n,length(t)-100);
% % F=zeros(2,n,length(t)-100);
% % for q=1:length(t)-100
% %     M1_(:,:,q)=[theta11(miu(1)+1:n+miu(1),q)'; theta21(miu(2)+1:n+miu(2),q)'];
% %     M2_(:,:,q)=[theta12(miu(1)+1:n+miu(1),q)'; theta22(miu(2)+1:n+miu(2),q)'];
% %     %F(:,:,q)=[theta11(miu(1)+n+1:2*n+miu(1),q)'; theta21(miu(2)+n+1:miu(2)+2*n,q)'];
% %     F(:,:,q)=[theta12(miu(1)+n+1:2*n+miu(1),q)'; theta22(miu(2)+n+1:miu(2)+2*n,q)'];
% % end
% % G=[G1 G2];
% % M_=[M1_ M2_];
% %........................................................................................
% %A~=1-2.564z^-1+2.2365z^-2-0.6725z^-3
% % According to the discrete transfer function, below parameters have been
% % defined
% na=3;
% nb1=1; nb2=1;
% nb=nb1;
% d=0;
% N1=d+1;
% N2=d+P;
% a_=[1 -2.564 2.2365 -0.6725];
% b1_=num1(2:end);
% b2_=num2(2:end);
% C=1;  % because of using white noise
% f=zeros(P+d,na+1);
% f(1,1:3)=-1*a_(2:4);
% for j=1:P+d-1
%     for i=1:na
%         f(j+1,i)=f(j,i+1)-f(j,1)*a_(i+1);
%     end
% end
% F=zeros(2,na);
% F(1,:)=f(miu(1),1:na);
% F(2,:)=f(miu(2),1:na);
% %.......................................
% E1=zeros(P);
% E1(:,1)=1;
% for j=1:P-1
%     E1(j+1:P,j+1)=f(j,1);
% end
% B1=zeros(P,P+nb);
% for k=1:P
%         B1(k,k:k+1)=b1_;
% end
% m1_=E1*B1;
% M1_=zeros(P,nb+d);
% for k=1:P
%     M1_(k,:)=m1_(k,k+1);
% end
% %............................
% E2=zeros(P);
% E2(:,1)=1;
% for j=1:P-1
%     E2(j+1:P,j+1)=f(j,1);
% end
% B2=zeros(P,P+nb);
% for k=1:P
%         B2(k,k:k+1)=b2_;
% end
% m2_=E2*B2;
% M2_=zeros(P,nb+d);
% for k=1:P
%     M2_(k,:)=m2_(k,k+1);
% end
% M1(1,:)=M1_(miu(1),:);
% M1(2,:)=M1_(miu(2),:);
% M2(1,:)=M2_(miu(1),:);
% M2(2,:)=M2_(miu(2),:);
% M_=[M1 M2];
% %................................................
% %.....................Toeplitz Matrix.................................
% b1 = zeros(1,P); b1(1,1)= g1(2);
% a1 = g1(2:P+1);
% G1 = toeplitz(a1,b1);
% G1(:,M) = G1(:,M:P)*ones(P-M+1,1);
% G1 = G1(:,1:M);
% %........................................................
% b2 = zeros(1,P); b2(1,1)= g2(2);
% a2 = g2(2:P+1);
% G2 = toeplitz(a2,b2);
% G2(:,M) = G2(:,M:P)*ones(P-M+1,1);
% G2 = G2(:,1:M);
% %......................................
% G11=[G1(miu(1),1) G1(miu(1),m1+1); G1(miu(2),1) G1(miu(2),m1+1)];
% G21=[G2(miu(1),1) G2(miu(1),m1+1); G2(miu(2),1) G2(miu(2),m1+1)];
%  G=[G11 G21];
% %....................................................................................................................
% gamma =1;
% gain_DC=(num1(1)+num1(2)+num1(3))/(den1(1)+den1(2)+den1(3));
% gain_DC2=(num2(1)+num2(2)+num2(3))/(den2(1)+den2(2)+den2(3));
% Q = eye(2);
% R1 =((1.2)^2)*gamma*gain_DC^2*eye(2);
% R2=gamma*gain_DC2^2*eye(2);
% R=[R1 zeros(2); zeros(2) R2];
% alpha=0.5;
% Kpfc=(G'*Q*G+R)\(G'*Q);
% x01=0.0882;
% x02=441.2;
% %.............................................
% dU1_=zeros(nb+d,length(t));
% dU2_=zeros(nb+d,length(t));
% dU_=[dU1_;dU2_];
% d1=zeros(1,length(t));
% %y1=0; %linear
% y1=441.2;
% u_1=[];
% u_2=[];
% ym=[];
% y=0;
% Y_d=zeros(2,length(t));
% Y_past=zeros(2,length(t));
% Y_m=zeros(2,length(t));
% D=zeros(2,length(t));
% E=zeros(2,length(t));
% dU1=zeros(2,length(t));
% dU2=zeros(2,length(t));
% dU=[dU1;dU2];
% U1=zeros(M,length(t));
% U2=zeros(M,length(t));
% Y_=zeros(na,length(t));
% %..................step...........................
% r =ones(length(t),1);
% %..................................................
% 
% for i=1:length(t)-1
%     
% for j=1:2
%   Y_d(j,i+1)=(alpha^j)*y+(1-(alpha)^j)*r(i+1); % Programmed
% end 
% 
% Y_past(:,i+1)=M_*dU_(:,i+1)+F*Y_(:,i+1);
% D(:,i+1)=d1(i+1)*ones(2,1);
% 
% E(:,i+1)=Y_d(:,i+1)-Y_past(:,i+1)-D(:,i+1);
% 
% dU(:,i+1)=Kpfc*E(:,i+1);
% dU1(:,i+1)=dU(1:2,i+1);
% dU2(:,i+1)=dU(3:4,i+1);
% U1(1,i+1)=dU1(1,i+1)+U1(1,i);
% U2(1,i+1)=dU2(1,i+1)+U2(1,i);
% dU(:,i+1)=[dU1(:,i+1);dU2(:,i+1)];
% 
% Y_m(:,i+1)=G*dU(:,i+1)+Y_past(:,i+1);
% 
% dU1_(2:nb+d,i+2) = dU1_(1:nb+d-1,i+1);
% dU1_(1,i+2)=dU1(1,i+1);
% dU2_(2:nb+d,i+2) = dU2_(1:nb+d-1,i+1);
% dU2_(1,i+2)=dU2(1,i+1);
% dU_(:,i+2)=[dU1_(:,i+2);dU2_(:,i+2)];
% % Y_(2:na,i+2)=Y_(1:na-1,i+1);
% % Y_(1,i+2)=Y_m(1,i+1);
% Y_(2:na,i+2+miu(1))=Y_(1:na-1,i+1+miu(1));
% Y_(1,i+2+miu(1))=Y_m(1,i+1);
% 
% u1=U1(1,i+1);
% u2=U2(1,i+1);
% sim('Model');
% %d(i+2)=yl(end)-Y_m(1,i); %linear
% d1(i+2)=y(end)-Y_m(1,i+1);
% %y=yl(end); % linear
% y=y(end);%+dist(i,1);    % nonlinear
% %y1=[y1;yl(end)];  % linear
% y1=[y1; y+441.2];
% ym=[ym; Y_m(1,i)];
% u_1=[u_1; u1];
% u_2=[u_2; u2];
% %noise=[noise; n];
% x01=x1(end);
% x02=x2(end);
% 
% end
% 
% figure(1);
% plot(y1);
% hold on
% plot(r+441.2,'r');
%--------------------------------------------------------------------------------------
%--------------------------------------------------------------------------------------
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
t=1:Ts:130;
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
           %theta1(:,j)=(phi1(:,j)'*phi1(:,j))\(phi1(:,j)*y(1,j));
           theta1(:,j)=pinv(phi1(:,j)')*y(1,j);
        else
            %theta2  (:,j)=(phi1(:,j)'*phi1(:,j))\(phi1(:,j)*y(1,j));
            theta2(:,j)=pinv(phi1(:,j)')*y(1,j);
        end
    end
end
%..........................................................................................................................................................
% G1=zeros(2,2,length(t)-100);
% G2=zeros(2,2,length(t)-100);
M1_=zeros(2,n,length(t)-100);
M2_=zeros(2,n,length(t)-100);
F=zeros(2,n,length(t)-100);
for q=1:length(t)-100
%     G1(:,:,q)=[theta1(miu(1),q) theta1(miu(1)-m1,q); theta2(miu(2),q) theta2(miu(2)-m1,q)];
%     G2(:,:,q)=[theta1(2*miu(1),q) theta1(2*miu(1)-m1,q); theta2(2*miu(2),q) theta2(2*miu(2)-m1,q)];
    M1_(:,:,q)=[theta1(2*miu(1)+1:n+2*miu(1),q)'; theta2(2*miu(2)+1:n+2*miu(2),q)'];
    M2_(:,:,q)=[theta1(2*miu(1)+1+n:2*n+2*miu(1),q)'; theta2(2*miu(2)+n+1:2*n+2*miu(2),q)'];
    F(:,:,q)=[theta1(2*miu(1)+2*n+1:3*n+2*miu(1),q)'; theta2(2*miu(2)+2*n+1:2*miu(2)+3*n,q)'];
end
%G=[G1 G2];
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
x01=0.0882;
x02=441.2;
dU1_=zeros(n,length(t)-100);
dU2_=zeros(n,length(t)-100);
dU_=[dU1_;dU2_];
d1=zeros(1,length(t)-100);
%y1=0; %linear
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
% dist=zeros(length(t),1);
% dist(200:249,1)=ones(50,1);
%..................step...........................
%r =ones(length(t)-100,1);
%...................sine..............................
%[r,t1]= gensig('sine',length(t)*Ts/2,length(t)*Ts,Ts);
%....................pulse............................
 [r,t1]= gensig('square',length(t)*Ts,length(t)*Ts,Ts);
%r=r;
 for l = 1:length(t)
 if (r(l)==0)
r(l) = -1;
 end
end
%..................................................

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
% Y_(2:n,i+2+miu(1))=Y_(1:n-1,i+1+miu(1));
% Y_(1,i+2+miu(1))=Y_m(1,i+1);

u1=U1(1,i+1);
u2=U2(1,i+1);
sim('Model');
%d(i+2)=yl(end)-Y_m(1,i); %linear
d1(i+2)=y(end)-yl(end);
Y_(2:n,i+2)=Y_(1:n-1,i+1);
Y_(1,i+2)=yl(end);
%y=yl(end); % linear
y=y(end);%+dist(i,1);    % nonlinear
%y1=[y1;yl(end)];  % linear
y1=[y1; y+441.2];
ym=[ym; yl(end)];
u_1=[u_1; u1];
u_2=[u_2; u2];
%noise=[noise; n];
x01=x1(end);
x02=x2(end);

end

figure(1);
subplot(2,2,1);
plot(y1,'b');
hold on
plot(r+441.2,'r');
 grid on
%axis([0 600 440 448]);
legend('y','r');
title('Response of the nonlinear system');
xlabel('sample');
%figure(4);
subplot(2,2,2);
plot(y1-441.2,'b');
hold on
plot(ym,'r');
grid on
xlabel('sample');
title('Ym and Yp without bias');
legend('YPlant','YModel');
%figure(5);
subplot(2,2,3);
plot(u_1,'b');
grid on
xlabel('sample');
title('Control law for input 1 without bias');
%figure(6);
subplot(2,2,4);
plot(u_2,'b');
grid on
xlabel('sample');
title('Control law for input 2 without bias');

%---------------------------------------------------------------------------------------------------------
%--------------------------------------------------------------------------------------------------------------
%------------------------------------------------------------------------------------------------------------------------------------------------
%----------------------------------------------- this part is completely IIR--------------------------------------------------------------
% clear
% clc
% [n1,d1,n2,d2]=Inputsys(1);
% Gs1 = tf(n1,d1);
% Ts=0.1;
% Gd1 = c2d(Gs1,Ts,'zoh');
% [num1,den1]=tfdata(Gd1,'v');
% Gs2 = tf(n2,d2);
% Gd2 = c2d(Gs2,Ts,'zoh');
% [num2,den2]=tfdata(Gd2,'v');
% sys_info = stepinfo(Gd1);
% ts1 = sys_info.SettlingTime;
% tr1=sys_info.RiseTime; 
% sys_info = stepinfo(Gd2);
% ts2 = sys_info.SettlingTime;
% tr2=sys_info.RiseTime;
% t=1:Ts:300;
% P1=floor(tr1/Ts);
% P2=floor(tr2/Ts);
% N1=floor( ts1/Ts);
% N2=floor( ts2/Ts);
% P=max(P1,P2);
% N=max(N1,N2);
% M=P;
% n=3;
% m1=2;
% miu=[5 6];
% ui1=randn(1,length(t));
% ui2=randn(1,length(t));
% ui=[ui1; ui2];
% Gd=[Gd1 Gd2];
% [yi,t1]=lsim(Gd,ui,t);
% theta1=zeros(2*miu(1)+3*n,length(t)-100);
% theta2=zeros(2*miu(2)+3*n,length(t)-100);
% for i=1:2
%     phi1=zeros(2*miu(i)+3*n,length(t));
%     for j=1:length(t)-100
%         y(1,j)=yi(j+miu(i));
%         for k=1:miu(i)
%             if (j+miu(i)-k-1)<=0
%                 du(j+miu(i)-k)=ui1(j+miu(i)-k);
%                 du2(j+miu(i)-k)=ui2(j+miu(i)-k);
%             else
%                 du(j+miu(i)-k)=ui1(j+miu(i)-k)-ui1(j+miu(i)-k-1);
%                 du2(j+miu(i)-k)=ui2(j+miu(i)-k)-ui2(j+miu(i)-k-1);
%             end
%             phi1(k,j)=du(j+miu(i)-k);
%             phi1(k+miu(i),j)=du2(j+miu(i)-k);
%         end
%         for k=1:n
%             if (j-k-1)<=0
%                 phi1(k+2*miu(i),j)=0;
%                 phi1(k+2*miu(i)+n,j)=0;
%             else
%                 phi1(k+2*miu(i),j)=ui1(j-k)-ui1(j-k-1);
%                 phi1(k+2*miu(i)+n,j)=ui2(j-k)-ui2(j-k-1);
%             end
%             if (j-k)<=0
%                  phi1(k+2*miu(i)+2*n,j)=0;
%             else
%                  phi1(k+2*miu(i)+2*n,j)=yi(j-k);  
%             end
%         end  
%         if i==1
%             %theta1(:,j)=pinv(phi1(:,j)*phi1(:,j)')*(phi1(:,j)*y(1,j));
%               %theta1(:,j)=(phi1(:,j)'*phi1(:,j))\(phi1(:,j)*y(1,j));
%               theta1(:,j)=pinv(phi1(:,j)')*y(1,j);
%         else
%             %theta2(:,j)=(phi1(:,j)'*phi1(:,j))\(phi1(:,j)*y(1,j));
%             %theta2(:,j)=pinv(phi1(:,j)*phi1(:,j)')*(phi1(:,j)*y(1,j));
%             theta2(:,j)=pinv(phi1(:,j)')*y(1,j);
%         end
%     end
% end
% %..........................................................................................................................................................
% G1=zeros(2,2,length(t)-100);
% G2=zeros(2,2,length(t)-100);
% M1_=zeros(2,n,length(t)-100);
% M2_=zeros(2,n,length(t)-100);
% F=zeros(2,n,length(t)-100);
% for q=1:length(t)-100
%     G1(:,:,q)=[theta1(miu(1),q) theta1(miu(1)-m1,q); theta2(miu(2),q) theta2(miu(2)-m1,q)];
%     G2(:,:,q)=[theta1(2*miu(1),q) theta1(2*miu(1)-m1,q); theta2(2*miu(2),q) theta2(2*miu(2)-m1,q)];
%     M1_(:,:,q)=[theta1(2*miu(1)+1:n+2*miu(1),q)'; theta2(2*miu(2)+1:n+2*miu(2),q)'];
%     M2_(:,:,q)=[theta1(2*miu(1)+1+n:2*n+2*miu(1),q)'; theta2(2*miu(2)+n+1:2*n+2*miu(2),q)'];
%     F(:,:,q)=[theta1(2*miu(1)+2*n+1:3*n+2*miu(1),q)'; theta2(2*miu(2)+2*n+1:2*miu(2)+3*n,q)'];
% end
% G=[G1 G2];
% M_=[M1_ M2_];
% %....................................................................................................................
% gamma =1;
% gain_DC=(num1(1)+num1(2)+num1(3))/(den1(1)+den1(2)+den1(3));
% gain_DC2=(num2(1)+num2(2)+num2(3))/(den2(1)+den2(2)+den2(3));
% Q = eye(2);
% R1 =((1.2)^2)*gamma*gain_DC^2*eye(2);
% R2=gamma*gain_DC2^2*eye(2);
% R=[R1 zeros(2); zeros(2) R2];
% alpha=0.5;
% for q=1:length(t)-100
%     Kpfc(:,:,q)=(G(:,:,q)'*Q*G(:,:,q)+R)\(G(:,:,q)'*Q);
% end
% x01=0.0882;
% x02=441.2;
% dU1_=zeros(n,length(t));
% dU2_=zeros(n,length(t));
% dU_=[dU1_;dU2_];
% d1=zeros(1,length(t));
% %y1=0; %linear
% y1=441.2;
% u_1=[];
% u_2=[];
% ym=[];
% y=0;
% Y_d=zeros(2,length(t));
% Y_past=zeros(2,length(t));
% Y_m=zeros(2,length(t));
% D=zeros(2,length(t));
% E=zeros(2,length(t));
% dU1=zeros(2,length(t));
% dU2=zeros(2,length(t));
% dU=[dU1;dU2];
% U1=zeros(M,length(t));
% U2=zeros(M,length(t));
% Y_=zeros(n,length(t));
% % dist=zeros(length(t),1);
% % dist(200:249,1)=ones(50,1);
% %..................step...........................
% r =ones(length(t),1);
% %...................sine..............................
% %[r,t1]= gensig('sine',length(t)*Ts/2,length(t)*Ts,Ts);
% %....................pulse............................
% %  [r,t1]= gensig('square',length(t)*Ts,length(t)*Ts,Ts);
% % %r=r;
% %  for l = 1:length(t)
% %  if (r(l)==0)
% % r(l) = -1;
% %  end
% % end
% %..................................................
% 
% for i=1:length(t)-101
%     
% for j=1:2
%   Y_d(j,i+1)=(alpha^j)*y+(1-(alpha)^j)*r(i+1); % Programmed
% end 
% 
% Y_past(:,i+1)=M_(:,:,i+1)*dU_(:,i+1)+F(:,:,i+1)*Y_(:,i+1);
% D(:,i+1)=d1(i+1)*ones(2,1);
% 
% E(:,i+1)=Y_d(:,i+1)-Y_past(:,i+1)-D(:,i+1);
% 
% dU(:,i+1)=Kpfc(:,:,i+1)*E(:,i+1);
% dU1(:,i+1)=dU(1:2,i+1);
% dU2(:,i+1)=dU(3:4,i+1);
% U1(1,i+1)=dU1(1,i+1)+U1(1,i);
% U2(1,i+1)=dU2(1,i+1)+U2(1,i);
% dU(:,i+1)=[dU1(:,i+1);dU2(:,i+1)];
% 
% Y_m(:,i+1)=G(:,:,i+1)*dU(:,i+1)+Y_past(:,i+1);
% 
% dU1_(2:n,i+2) = dU1_(1:n-1,i+1);
% dU1_(1,i+2)=dU1(1,i+1);
% dU2_(2:n,i+2) = dU2_(1:n-1,i+1);
% dU2_(1,i+2)=dU2(1,i+1);
% dU_(:,i+2)=[dU1_(:,i+2);dU2_(:,i+2)];
% % Y_(2:n,i+2)=Y_(1:n-1,i+1);
% % Y_(1,i+2)=Y_m(1,i+1);
% % Y_(2:n,i+2+miu(1))=Y_(1:n-1,i+1+miu(1));
% % Y_(1,i+2+miu(1))=Y_m(1,i+1);
% 
% u1=U1(1,i+1);
% u2=U2(1,i+1);
% sim('Model');
% %d(i+2)=yl(end)-Y_m(1,i); %linear
% %d1(i+2)=y(end)-Y_m(1,i+1);
% d1(i+2)=y(end)-yl(end);
% %y=yl(end); % linear
% y=y(end);%+dist(i,1);    % nonlinear
% %y1=[y1;yl(end)];  % linear
% Y_(2:n,i+2)=Y_(1:n-1,i+1);
% Y_(1,i+2)=yl(end);
% y1=[y1; y+441.2];
% ym=[ym; Y_m(1,i)];
% u_1=[u_1; u1];
% u_2=[u_2; u2];
% %noise=[noise; n];
% x01=x1(end);
% x02=x2(end);
% 
% end
% 
% figure(1);
% plot(y1,'b');
% hold on
% plot(r+441.2,'r');

%---------------------------------------------------------------------------
    



    


    
