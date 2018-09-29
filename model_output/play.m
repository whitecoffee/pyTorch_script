figure()
plot3(ydisplay(121:160,1),ydisplay(121:160,2),ydisplay(121:160,3))
hold on
grid on
plot3(ydisplayval(121:157,1),ydisplayval(121:157,2),ydisplayval(121:157,3))
%plot3(fc(121:160,1),fc(121:160,2),fc(121:160,3))
%%
figure()
plot(fc(121:160,1))
hold on
grid on
plot(ydisplayval(121:160,1))

%%
figure()
plot(fc(121:160,2))
hold on
grid on
plot(ydisplayval(121:160,2))

%%
figure()
plot(fc(121:160,3))
hold on
grid on
plot(ydisplayval(121:160,3))