%{
for i=1:18
    a = out(i, :, :);
    a = reshape(a, [90, 3]);
    plot3(a(:,1), a(:,2), a(:,3));
    hold on;
end
%}
%{
bad
for i=19:26
    a = out(i, :, :);
    a = reshape(a, [90, 3]);
    plot3(a(:,1), a(:,2), a(:,3));
    hold on;
end
%}
%{
bad
for i=27:32
    a = out(i, :, :);
    a = reshape(a, [90, 3]);
    plot3(a(:,1), a(:,2), a(:,3));
    hold on;
end
%}
for i=33:40
    a = out(i, :, :);
    a = reshape(a, [90, 3]);
    plot3(a(:,1), a(:,2), a(:,3));
    hold on;
end

%{
for i=40:68
    a = out(i, :, :);
    a = reshape(a, [90, 3]);
    plot3(a(:,1), a(:,2), a(:,3));
    hold on;
end
%}

%{
a = out(65, :, :);
a = reshape(a, [90, 3]);
plot3(a(:,1), a(:,2), a(:,3));
hold on;
%}
