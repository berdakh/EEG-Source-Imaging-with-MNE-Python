%% find allfiles with '.mat' and process in batch and save the results afterwards
a = dir('*.mat');
len = length(a);

for ii = 1:len    
    filename = a(ii).name;    
    disp(filename)    
    load(filename)
    
    subject = filename;    
    experiment = 'MRCP';
    label = 'pre-processed';
    session = '1';
    fsample = 500;

    capFile = '64chan'; %[str] file to get electrode position info into
    [Cname,ra,xy,xyz] = readCapInf(capFile);

    di=mkDimInfo(size(X),'ch',[],[],'time','ms',[],'epoch',[],[]);
    %*************************************************
    z=jf_import(experiment,subject, label, X ,di,'fs', fsample,...
        'Cnames',Cname,'capFile',capFile,'session',session);
    
    % Remove bad epochs
    z = jf_rmOutliers(z,'dim','epoch','thresh', 2.5);
    % Linear detrending
    z = jf_detrend(z,'order',1,'dim','time');
    % Rereferencing using Common Average Referencing
    z = jf_reref(z,'dim','ch');
    % Remove bad channels and rebuild them with Spherical Spline Interpolation
    z = jf_rmOutliers(z,'dim','ch','thresh', 2.5);
    z = jf_spatdownsample(z,'capFile',capFile,'method','sphericalSplineInterpolate');
    
    % Filter spectrally
    z = jf_fftfilter(z,'fs',fsample,'bands',[0.1 4],'verb',-1);    
    X = z.X;    
    save(strcat(filename(1:end-8), '.mat'), 'X')  
    clear X
end 