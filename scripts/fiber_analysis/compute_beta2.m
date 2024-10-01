clear all;
close all;
load neffSI_smooth_1400_1600.mat
lambda = [1400:10:1600];
neffMatrix = zeros(12,length(lambda));

function [fitresult, gof] = determine_birefringence_fit(f, beta,norder)
    %CREATEFIT(X,Y)
    %  Create a fit.
    %
    %  Data for 'untitled fit 1' fit:
    %      X Input: x
    %      Y Output: y
    %  Output:
    %      fitresult : a fit object representing the fit.
    %      gof : structure with goodness-of fit info.
    %
    %  See also FIT, CFIT, SFIT.
    
    %  Auto-generated by MATLAB on 19-Feb-2024 18:36:36
    if nargin==2
        norder = 2;
    end    
    %% Fit: 'untitled fit 1'.
    
    [xData, yData] = prepareCurveData( f, beta );
    
    % Set up fittype and options.
    if norder==3
        ft = fittype( 'poly3' );
    else
        ft = fittype( 'poly2' );
    end
    
    % Fit model to data.
    [fitresult, gof] = fit( xData, yData, ft, 'Normalize', 'off' );
    
    % % Plot fit with data.
    % figure( 'Name', 'untitled fit 1' );
    % h = plot( fitresult, xData, yData );
    % legend( h, 'y vs. x', 'untitled fit 1', 'Location', 'NorthEast', 'Interpreter', 'none' );
    % % Label axes
    % xlabel( 'x', 'Interpreter', 'none' );
    % ylabel( 'y', 'Interpreter', 'none' );
    % grid on
end

%% Run
for i = 16:-1:5
    neffMatrix(17-i,:)=neffSIsmooth14001600(i:14:end);
end

omega = physconst('LightSpeed')./lambda * 1e9 * 2*pi;
betaMatrix = omega.*neffMatrix/physconst('LightSpeed'); %m^-1

betaMatrix=betaMatrix';
unitfactor = 1e27; % ps^2/km
omega_n = (omega - mean(omega))./std(omega)
% for i = 1:12
%     [fitresult, gof] = determine_birefringence_fit(omega_n, betaMatrix(:,i),2)
%     beta2_2(i) = 2 * fitresult.p1./std(omega).^2 * unitfactor;
%     beta1_2(i,:) = (2 * fitresult.p1.*omega_n + fitresult.p2)./std(omega);
% end


% for i = 1:12
%     [fitresult, gof] = determine_birefringence_fit(omega, betaMatrix(:,i))
%     beta2(i) = 2 * fitresult.p1;
%     beta1(i,:) = 2 * fitresult.p1.*omega + fitresult.p2;
% end

for i = 1:12
    [fitresult, gof] = determine_birefringence_fit(omega_n, betaMatrix(:,i),3);
    beta1_3(i,:) = (3 * fitresult.p1.*(omega_n).^2 + 2 * fitresult.p2.*omega_n +fitresult.p3)./std(omega);
    beta2_3(i,:) = (6 * fitresult.p1.*(omega_n) + 2 * fitresult.p2)./std(omega).^2*unitfactor;
end

betaMatrix_avg = zeros(size(betaMatrix, 1), 4);
betaMatrix_avg(:, 1) = mean(betaMatrix(:, 1:2), 2);
betaMatrix_avg(:, 2) = mean(betaMatrix(:, 3:6), 2);
betaMatrix_avg(:, 3) = mean(betaMatrix(:, 7:10), 2);
betaMatrix_avg(:, 4) = mean(betaMatrix(:, 11:12), 2);

%%%% Already averaged results
fitParams = zeros(4, 3)
for i = 1:4
    [fitresult, gof] = determine_birefringence_fit(omega_n, betaMatrix_avg(:,i),3);
    fitParams(i,:) = [fitresult.p1, fitresult.p2, fitresult.p3];
end

meanM = [ones(1,2)/2, zeros(1,10);...
    zeros(1,2),ones(1,4)/4,zeros(1,6);...
    zeros(1,6),ones(1,4)/4,zeros(1,2);...
    zeros(1,10),ones(1,2)/2]';

mean_beta1 = beta1_3'*meanM;


figure;
plot(omega/2/pi*1e-12,10^12*(mean_beta1(:,2:end)-mean_beta1(:,1)),'LineWidth',1.5)
xlabel('f (THz)');
ylabel('DMGD (ps/m)')
grid on;
legend('LP_{1,1}-LP_{0,1}','LP_{2,1}-LP_{0,1}','LP_{0,2}-LP_{0,1}');
saveas(gcf,'DMGD.png');
figure;
plot(omega/2/pi*1e-12,beta1_3'*meanM*10^9,'LineWidth',1.5)
xlabel('f (THz)');
ylabel('\beta_1 (us/km)')
grid on;
legend('LP_{0,1}',...
    'LP_{1,1}',...
    'LP_{2,1}',...
    'LP_{0,2}');
% saveas(gcf,'beta2.png')
% legend('LP_{0,1}','LP_{0,1}',...
%     'LP_{1,1}','LP_{1,1}','LP_{1,1}','LP_{1,1}',...
%     'LP_{2,1}','LP_{2,1}','LP_{2,1}','LP_{2,1}',...
%     'LP_{0,2},LP_{0,2}');

% Save fit parameters to MAT file
fitParams
omega_std = std(omega)
omega_mean = mean(omega)
save('../../results/fitBeta.mat', ['fitParams'], 'omega_std', 'omega_mean');