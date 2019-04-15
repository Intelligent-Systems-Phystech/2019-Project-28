function histo(matr,feat_name,segments,name_data)

num_feat=size(matr,2);
num_obj=size(matr,1);
% segments=20;

for i=1:num_feat
    
vec=zeros(segments,1);
step=max(matr(:,i))-min(matr(:,i));
x=linspace(min(matr(:,i)),max(matr(:,i)),segments);
for j=1:segments
   
    vec(j,1)=size(find(matr(:,i)<=step*j/segments+min(matr(:,i))),1);
    
end

j=segments;
while (j>=2)
   
    vec(j,1)=vec(j,1)-vec(j-1,1);
    j=j-1;
    
end
    
h = figure; 
bar(x, vec/num_obj,'stack')
grid on
set(gca,'Layer','top') % display gridlines on top of graph
xlabel(feat_name{i},'FontSize',24,'FontName','Times','Interpreter','latex');
ylabel('Share of samples','FontSize',24,'FontName','Times','Interpreter','latex');
set(gca, 'FontSize', 20, 'FontName', 'Times');

name_prepared=strcat('histo_auc_',name_data);
saveas(h,strcat(name_prepared,'.eps'), 'psc2');
saveas(h,strcat(name_prepared,'.png'), 'png');
end

end