%% Set up the path, retrieve list of cells to do
Path2Project = '/Users/elie/Documents/ManipBats/PlaneCells';
addpath('/Users/elie/Documents/CODE/tlab/src/slurmbot/matlab')
% Retrieve list of cells
Data = load(fullfile(Path2Project,'data_table_index_for_shuffling_in_the_cluster.mat'));

%% Set up the variables for slurm
JobParams = struct;
JobParams.Partition = 'savio';
JobParams.Account = 'fc_batvl';
JobParams.Qos = 'savio_normal';
JobParams.NTasks = 1;
JobParams.CPU = 20;
JobParams.TimeLimit = '15:00:00';
JobParams.Name = 'PlaneCells';
SlurmParams.cmd = 'shuffle_single_cell_fn(''%s'', ''%s'',''%s'');';
SlurmParams.resultsDirectory='/global/scratch/jelie/PlaneCells';

%% Set up variables to identify cells to run and order
NumCells = size(Data.data_index_table,1);

%% Create jobs' files
cd(fullfile(Path2Project, 'Job2DoSavio'))
for ff=1:NumCells
    Batname = Data.data_index_table{ff,1};
    Date = Data.data_index_table{ff,2};
    CellNum = Data.data_index_table{ff,3};
    fprintf(1,'creating job file %d/%d: %s %s %s\n',ff,NumCells, Batname, Date,CellNum);
    JobParams.JobName = sprintf('%s_%s_%s',Batname,Date,CellNum);
    JobParams.out = fullfile(SlurmParams.resultsDirectory,sprintf('slurm_out_%s_%%j.txt', JobParams.JobName));
    JobParams.err = JobParams.out;
    icmd = sprintf(SlurmParams.cmd, Batname, Date,CellNum);
    fprintf(1,'creating file slurm_sbatch with command %s\n',icmd);
    slurm_sbatch_savio(icmd,JobParams); 
end

    
fprintf(1,'DONE Creating all Jobs'' files!!!\n');