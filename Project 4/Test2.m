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
% t=1:Ts:200;
% P1=floor(tr1/Ts);
% P2=floor(tr2/Ts);
% N1=floor( ts1/Ts);
% N2=floor( ts2/Ts);
% P=max(P1,P2);
% N=max(N1,N2);
% M=P;
% n=3;
% m1=2;
% miu=[5 10];
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
%             theta1(:,j)=(phi1(:,j)'*phi1(:,j))\(phi1(:,j)'*y(1,j));
%         else
%             theta2(:,j)=(phi1(:,j)'*phi1(:,j))\(phi1(:,j)'*y(1,j));
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
%.................................................................................................................................................................
%.............................................................................................................................................................
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
% t=1:Ts:200;
% P1=floor(tr1/Ts);
% P2=floor(tr2/Ts);
% N1=floor( ts1/Ts);
% N2=floor( ts2/Ts);
% P=max(P1,P2);
% N=max(N1,N2);
% M=P;
% n=3;
% m1=2;
% miu=[5 10];
% theta11=zeros(miu(1)+2*n,length(t)-100);
% theta21=zeros(miu(2)+2*n,length(t)-100);
% for i=1:2
%     ui=randn(1,length(t));
%     [yi,t1]=lsim(Gd1,ui,t);
%     phi1=zeros(miu(i)+2*n,length(t));
%     for j=1:length(t)-100
%         y(1,j)=yi(j+miu(i));
%         for k=1:miu(i)
%             if (j+miu(i)-k-1)<=0
%                 du(j+miu(i)-k)=ui(j+miu(i)-k);
%             else
%                 du(j+miu(i)-k)=ui(j+miu(i)-k)-ui(j+miu(i)-k-1);
%             end
%             phi1(k,j)=du(j+miu(i)-k);
%         end
%         for k=1:n
%             if (j-k-1)<=0
%                 phi1(k+miu(i),j)=0;
%             else
%                 phi1(k+miu(i),j)=ui(j-k)-ui(j-k-1);
%             end
%             if (j-k)<=0
%                  phi1(k+miu(i)+n,j)=0;
%             else
%                  phi1(k+miu(i)+n,j)=yi(j-k);  
%             end
%         end  
%         if i==1
%             theta11(:,j)=(phi1(:,j)'*phi1(:,j))\(phi1(:,j)'*y(1,j));
%         else
%             theta21(:,j)=(phi1(:,j)'*phi1(:,j))\(phi1(:,j)'*y(1,j));
%         end
%     end
% end
% %..........................................................................................................................................................
% theta12=zeros(2*n+miu(1),length(t)-100);
% theta22=zeros(2*n+miu(2),length(t)-100);
% for i=1:2
%     ui=randn(1,length(t));
%     [yi2,t1]=lsim(Gd2,ui,t);
%     phi2=zeros(2*n+miu(i),length(t));
%     for j=1:length(t)-100
%         y2(1,j)=yi2(j+miu(i));
%         for k=1:miu(i)
%             if (j+miu(i)-k-1)<=0
%                 du(j+miu(i)-k)=ui(j+miu(i)-k);
%             else
%                 du(j+miu(i)-k)=ui(j+miu(i)-k)-ui(j+miu(i)-k-1);
%             end
%             phi2(k,j)=du(j+miu(i)-k);
%         end
%         for k=1:n
%             if (j-k-1)<=0
%                 phi2(k+miu(i),j)=0;
%             else
%                 phi2(k+miu(i),j)=ui(j-k)-ui(j-k-1);
%             end
%             if (j-k)<=0
%                  phi2(k+n+miu(i),j)=0;
%             else
%                  phi2(k+n+miu(i),j)=yi2(j-k);  
%             end
%         end
%         if i==1
%             theta12(:,j)=(phi2(:,j)'*phi2(:,j))\(phi2(:,j)'*y2(1,j));
%         else
%             theta22(:,j)=(phi2(:,j)'*phi2(:,j))\(phi2(:,j)'*y2(1,j));
%         end
%     end
% end
% G1=zeros(2,2,length(t)-100);
% G2=zeros(2,2,length(t)-100);
% M1_=zeros(2,n,length(t)-100);
% M2_=zeros(2,n,length(t)-100);
% F=zeros(2,n,length(t)-100);
% for q=1:length(t)-100
%     G1(:,:,q)=[theta11(miu(1),q) theta11(miu(1)-m1,q); theta21(miu(2),q) theta21(miu(2)-m1,q)];
%     G2(:,:,q)=[theta12(miu(1),q) theta12(miu(1)-m1,q); theta22(miu(2),q) theta22(miu(2)-m1,q)];
%     M1_(:,:,q)=[theta11(miu(1)+1:n+miu(1),q)'; theta21(miu(2)+1:n+miu(2),q)'];
%     M2_(:,:,q)=[theta12(miu(1)+1:n+miu(1),q)'; theta22(miu(2)+1:n+miu(2),q)'];
%     %F(:,:,q)=[theta11(miu(1)+n+1:2*n+miu(1),q)'; theta21(miu(2)+n+1:miu(2)+2*n,q)'];
%     F(:,:,q)=[theta12(miu(1)+n+1:2*n+miu(1),q)'; theta22(miu(2)+n+1:miu(2)+2*n,q)'];
% end
% G=[G1 G2];
% M_=[M1_ M2_];