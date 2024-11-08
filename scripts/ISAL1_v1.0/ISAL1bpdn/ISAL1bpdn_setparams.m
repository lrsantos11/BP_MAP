function [dLBmode,dspl,maxitPr,p,stag,time,tolPr,m,n,scale,reset,sparseA,CGmode,x0,useHOC] = ISAL1bpdn_setparams(A,b,delta,varargin)
%function [dLBmode,dspl,maxitPr,p,stag,time,tolPr,m,n,scale,reset,sparseA,CGmode,x0,useHOC] = ISAL1bpdn_setparams(A,b,delta,varargin)
% =========================================================================
%  FILE:           ISAL1bpdn_setparams.m                                            
%                  ( This function is part of the ISAL1bpdn program, v1.0 )
%  AUTHOR:         Andreas M. Tillmann, TU Darmstadt, Germany             
%  LAST MODIFIED:  09/28/2013                                                                                              
% =========================================================================
%  DESCRIPTION:    This function initializes the parameter values for the
%                  infeasible-point subgradient method "ISAL1bpdn", 
%                  either with defaults or with user-supplied values. 
%                  User-specified values are checked for consistency, if
%                  something is not right (wrong class or dimensions), the
%                  default parameter settings are used. If decimal numbers
%                  are passed for integer arguments, the nearest integer
%                  value is used instead.
%
%                  Every user-specified argument must have been passed in 
%                  the form    'parameter name', value 
%                  thus every second entry of varargin{:}, starting from 
%                  the first one, is a parameter name, and each parameter 
%                  name is followed by the corresponding parameter value.
%
%                  Description of feasible parameter values can be found
%                  below (see 'parnames'-Initialization ).
%
%                  Copyright (C) Andreas M. Tillmann, 2013
%
% NOTE:            The implementation of ISAL1bpdn is a prototype, and under
%                  development; most parameters and settings are still 
%                  the same as for the pure BP version (ISAL1) and have 
%                  not been optimized w.r.t. solving BP-Denoising yet.
%                  (note, e.g., that delta is unused in the present file)
% =========================================================================

% Quick return:
if( isempty(A) || isempty(b) )
    fprintf('Error: Matrix A or vector b is empty.\n');
    [dLBmode,dspl,maxitPr,p,stag,time,tolPr,m,n,scale,reset,sparseA,CGmode,x0,useHOC] = deal([]);
    return;
elseif( isempty(varargin) ) % No user-specified arguments, use defaults    
    sparseA = issparse(A); 
    nnzA = nnz(A);
    [m, n] = size(A);
    numelA = m*n;   
    nnzdivnumel = nnzA/numelA;
    if( nnzdivnumel <= 0.1 )
        p = ceil( max( 5, n/(m*(nnzdivnumel)^(3/4)) ) );
    else
        p = ceil( max( 5, n/(m*(nnzdivnumel)^2) ) );  
    end
    CGmode = ((nnzA<0.1*m^2) || (sparseA)); 
        % computing x=A*A'*z as y=A'*z; x=A*y; should be faster than 
        % computing A*A'*z directly (not counting computation cost for A*A')

    if( (~sparseA) && (~CGmode) )         % dense: compute A*A' explicitly
        opts.SYM = true; opts.POSDEF = true;  % options for linsolve-Funktion                          
        CGmode = {CGmode,A*A',opts};          % --> both passed within 'CGmode'
        % clear opts;
%    else % A has sparse-format, or dense-format but is (relatively) sparse)
%        if( CGmode )   % sparse or dense (but relatively sparse)
%	     % nothing to do: A*A' is not needed, CG uses only A
%        else           % sparse, but CGmode=0: actually not possible in default setup!
%            CGmode = {CGmode,A*A'}; % CG is used, but with explicit A*A' matrix
%        end        
    end
    x0 = A'*(b./norm(b));  % default: scaled r.h.s.
    [dLBmode,dspl,maxitPr,stag,time,tolPr,scale,reset,useHOC] = deal(-1,inf,20,max(5*p,500),300,0.85,1,1,1);
    return;
end

% Available parameter names:
parnames = {'dlbmode';  % integer in [0,6], or negative number (turn off) 
                        %     - specifies how lower bounds are computed, see lowbo.m
            'dspl';     % integer in (0,inf]
                        %     - specifies at which iteration intervals information is displayed,
                        %       (0=only init. and final, 1=every iter., inf=no display at all)
            'maxitpr';  % integer (0,inf]    
                        %     - maximum number of iterations for projection algorithm
            'p';        % integer in (0,inf) 
                        %     - number of iterations w/o relevant obj. improvement after which
                        %       the stepsize is halved                        
            'stag';     % integer in (0,inf)
                        %     - number of iterations w/o relevant obj. improvement after which
                        %       ISAL1 terminates due to stagnation of algorithmic progress
            'time';     % integer in (0,inf]
                        %     - running time limit for ISAL1
            'tolpr';    % double in [0,1]
	                    %     - termination tolerance parameter for projection algorithm
            'scale';    % Boolean 
                        %     - value 1/true indicates scaling the r.h.s. to Eucl. norm 1
            'reset';    % Boolean
 	                    %     - toggle automatic restart (activated for value 1/true)
            'sparsea';  % Boolean
                        %     - true iff A is in sparse format
            'cgmode';   % binary
                        %     - specifies whether A*A' is to be computed explicitly (value 0) or
                        %       whether CG uses only A (value 1)
            'x0';       % nx1 double array (column vector) giving the starting point x0, or string 
                        % 'LS'  for Least-Squares solution to Ax=b as x0, or 
                        % 'P0'  for projection of all-zero vector onto {x:||Ax-b||<=delta}), or
                        % 'R'   for projection onto {x:||Ax-b||<=delta} of a (uniformly distr.) random vector
                        %       (multiplied by norm(b)), or
                        % 'ATb' for x0=A'*b (b possibly scaled to norm 1)
  	                    %       (theoretically it is then possible that ||x0||_1 < optimal value)
	        'usehoc'};  % Boolean
  	                    %     - toggle use of HOC termination criterion (active for value 1/true)
sparseA = issparse(A);                       
nnzA = nnz(A);
[m, n] = size(A);
numelA = m*n;       
nnzdivnumel = nnzA/numelA;
if( nnzdivnumel <= 0.1 )
    p = ceil( max( 5, n/(m*(nnzdivnumel)^(3/4)) ) );
else
    p = ceil( max( 5, n/(m*(nnzdivnumel)^2) ) );
end

% Set up default parameter values; user-supplied values are included later
parvalues = {-1;            % default: do not use ISAL1bpdn_lowbo.m, compute dLB via HOC
             inf;           % silent (no onscreen output)
             20;            % initial iter. limit for projection algorithm
             p;             % see above 
             max(500,5*p);  % stagnation number
             300;           % 5 minutes default time limit
             0.85;          % initial projection accuracy tolerance
             1;             % scale r.h.s.
             1;             % restart mechanism activated
             sparseA;       % true/1 iff A is stored in sparse format
             ((nnzA<0.1*m^2) || (sparseA));  % CGmode
             [];            % computation of starting point later, if necessary
             1};            % default: use HOC

% Identify the names of user-supplied parameters and set their values:         
for i=1:2:numel(varargin)
    j = strmatch(lower(varargin{i}),parnames,'exact');%find(strcmpi(varargin({i}),parnames));%recommend but not working!
    if( ~isempty(j) ) % Feasible argument name
        switch parnames{j}
            case 'dlbmode', 
                if( isempty(varargin{i+1}) )
                    fprintf('Argument dlbmode may not be empty-matrix. Default setting used.\n');
                elseif( ~isnumeric(varargin{i+1}) )
                    fprintf('Argument dlbmode has the wrong class (must be numeric). Default setting used.\n');
                elseif( numel(varargin{i+1})~=1 )
                    fprintf('Argument dlbmode has unmatching dimensions. Default setting used.\n');   
                elseif( round(varargin{i+1})>6 )
                    fprintf('Argument dlbmode is out of range [ 0,6 ]. Default setting used.\n');
                else
                    if( varargin{i+1} < 0 )
                        parvalues{j} = -1;
                    else
                        parvalues{j} = round(varargin{i+1});
                    end
                end
            case {'dspl','maxitpr','time'}, 
                if( isempty(varargin{i+1}) )
                    fprintf('Argument %s may not be empty-matrix. Default setting used.\n',parnames{j});
                elseif( ~isnumeric(varargin{i+1}) )
                    fprintf('Argument %s has the wrong class (must be numeric). Default setting used.\n',parnames{j}); 
                elseif( numel(varargin{i+1})~=1 )
                    fprintf('Argument %s has unmatching dimensions. Default setting used.\n',parnames{j});              
                elseif( varargin{i+1}<0 )
                    fprintf('Argument %s is out of range ( 0,inf ]. Default setting used.\n',parnames{j}); 
                else
                    parvalues{j} = round(varargin{i+1});
                end
            case {'p','stag'}, 
                if( isempty(varargin{i+1}) )
                    fprintf('Argument %s may not be empty-matrix. Default setting used.\n',parnames{j});
                elseif( ~isnumeric(varargin{i+1}) )
                    fprintf('Argument %s has the wrong class (must be numeric). Default setting used.\n',parnames{j});
                elseif( numel(varargin{i+1})~=1 )
                    fprintf('Argument %s has unmatching dimensions. ', ...
                        parnames{j}); fprintf('Default setting used.\n');                       
                elseif( round(varargin{i+1})<0 || isinf(varargin{i+1}) )
                    fprintf('Argument %s is out of range ( 0,inf ). ', ...
                        parnames{j}); fprintf('Default setting used.\n');
                else
                    parvalues{j} = round(varargin{i+1});
                end
            case {'tolpr'}, 
                if( isempty(varargin{i+1}) )
                    fprintf('Argument %s may not be empty-matrix. Default setting used.\n',parnames{j});
                elseif( ~isnumeric(varargin{i+1}) )
                    fprintf('Argument %s has the wrong class (must be double). Default setting used.\n',parnames{j});
                elseif( numel(varargin{i+1})~=1 )
                    fprintf('Argument %s has unmatching dimensions. Default setting used.\n',parnames{j});  
                elseif( (varargin{i+1}<0) || (varargin{i+1}>1) )
                    fprintf('Argument %s is out of range [ 0,1 ]. Default setting used.\n',parnames{j});
                else
                    parvalues{j} = varargin{i+1};
                end
            case {'reset','scale','cgmode','usehoc'},
                if( isempty(varargin{i+1}) )
                    fprintf('Argument %s may not be empty-matrix. Default setting used.\n',parnames{j});
                elseif( (~isnumeric(varargin{i+1})) && (~islogical(varargin{i+1})) )
                    fprintf('Argument %s has the wrong class (must be numeric or logical). Default setting used.\n',parnames{j}); 
                elseif( numel(varargin{i+1})~=1 )
                    fprintf('Argument %s has unmatching dimensions. Default setting used.\n',parnames{j});
                elseif( (varargin{i+1}<0) || (varargin{i+1}>1) )
                    fprintf('Argument %s is out of range [ 0,1 ]. Default setting used.\n',parnames{j});
                elseif( isnumeric(varargin{i+1}) )
                    parvalues{j} = logical(round(varargin{i+1}));
                else % islogical
                    parvalues{j} = varargin{i+1};
                end
            case 'x0',
                if( isempty(varargin{i+1}) )
                    fprintf('Argument x0 may not be empty-matrix. Default setting used.\n');
                elseif( ischar(varargin{i+1}) )
                    if( strcmp('LS',varargin{i+1}) )
                        parvalues{j} = A\b; % Least Squares Sol. to Ax=b
                    elseif( strcmp('P0',varargin{i+1}) )
                        % Projection of 0 onto the set {x|Ax=b}
                        if( (~sparseA) && (~parvalues{11}) )                            
                            opts.SYM = true; opts.POSDEF = true;
                            parvalues{11} = {parvalues{11},A*A',opts}; % adapt CGmode (see Quick Return above)
                            if( parvalues{8} ) % scale r.h.s. to norm 1
                                [temp,~] = linsolve(parvalues{11}{2},b./norm(b),opts);                        
                            else
                                [temp,~] = linsolve(parvalues{11}{2},b,opts);
                            end
                            parvalues{j} = A'*temp;
                            % clear temp opts;
                        else
                            if( parvalues{8} ) % scale r.h.s. to norm 1
                                if( parvalues{11} ) % CGmode=1
                                    temp = CG(A,[],b./norm(b),1e-12,m,parvalues{11});
                                else                % CGmode=0
                                    parvalues{11} = {parvalues{11},A*A'};
                                    temp = CG([],parvalues{11}{2},b./norm(b),1e-12,m,parvalues{11}{1});
                                end
                            else
                                if( parvalues{11} ) % CGmode=1
                                    temp = CG(A,[],b,1e-12,m,parvalues{11});
                                else                % CGmode=0
                                    parvalues{11} = {parvalues{11},A*A'};
				    temp = CG([],parvalues{11}{2},b,1e-12,m,parvalues{11}{1});
                                end
                            end
                            parvalues{j} = A'*temp;
                            % clear temp ;
                        end
                    elseif( strcmp('R',varargin{i+1}) )
                        % projection of a random point onto the constraint set         
                        %RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock))); % not available in Matlab R2013a anymore
                        rng('shuffle');
                        if( parvalues{8} ) % scale r.h.s. ?
                            point = rand(n,1)-0.5; 
                            if( (~sparseA) && (~parvalues{11}) )
                                opts.SYM = true; opts.POSDEF = true;     
                                parvalues{11} = {parvalues{11},A*A',opts};
                                [temp,~] = linsolve(parvalues{11}{2},A*point-b./norm(b),opts);                        
                                parvalues{j} = point-A'*temp;
                                % clear temp opts point;
                            else
                                if( parvalues{11} ) % CGmode=1
                                    temp = CG(A,[],A*point-b./norm(b),1e-12,m,parvalues{11});
                                else                % CGmode=0
                                    parvalues{11} = {parvalues{11},A*A'};
                                    temp = CG([],parvalues{11}{2},A*point-b./norm(b),1e-12,m,parvalues{11}{1});
                                end
                                parvalues{j} = point-A'*temp;
                                % clear temp point;
                            end
                        else
                            point = norm(b)*(rand(n,1)-0.5);
                            if( ~sparseA && ~parvalues{11} )                        
                                opts.SYM = true; opts.POSDEF = true;    
                                parvalues{11} = {parvalues{11},A*A',opts};            
                                [temp,~] = linsolve(parvalues{11}{2},A*point-b,opts);                        
                                parvalues{j} = point-A'*temp;
                                % clear temp opts point;
                            else
                                if( parvalues{11} ) % CGmode=1
                                    temp = CG(A,[],A*point-b./norm(b),1e-12,m,parvalues{11});
                                else                % CGmode=0
                                    parvalues{11} = {parvalues{11},A*A'};
                                    temp = CG([],parvalues{11}{2},A*point-b./norm(b),1e-12,m,parvalues{11}{1});
                                end
                                parvalues{j} = point-A'*temp;
                                % clear temp point;
                            end
                        end
                    elseif( strcmp('ATb',varargin{i+1}) )
                        % A'*b (feasible iff A*A' = I (partial isometry))                        
                        % this is the default and will be computed below                       
                    else
                        fprintf('Unknown choice for x0. Default setting used.\n');
                    end                    
                elseif( ~isnumeric(varargin{i+1}) )
                    fprintf('Argument x0 has the wrong class (must be numeric). Default setting used.\n');
                elseif( ~all(sort(size(varargin{i+1}))==[1 n]) )
                    fprintf('Argument x0 has unmatching dimensions. Default setting used.\n');   
                else
                    parvalues{j} = varargin{i+1}(:); % ensure x0 is a column vector
                    % adapt CGmode if a vector x0 was given (i.e., no predefined construction for x0 was chosen):
                    if( (~sparseA) && (~parvalues{11}) )    % dense-format, CGmode=0
                        opts.SYM = true; opts.POSDEF = true;     
                        parvalues{11} = {parvalues{11},A*A',opts};
                        % clear opts;
                    else
                        if( ~parvalues{11} ) % CGmode=0, but sparse A
                            parvalues{11} = {parvalues{11},A*A'};
                        %else  % CGmode=1 (dense or sparse)
                        %    % nothing to do
                        end                        
                    end
                end
        end
    end
end

% No (valid) starting point supplied by user, set up default x0 now and adapt CGmode:
if( isempty(parvalues{12}) )
    % default is A'*b (possibly w/ scaled b) (feasible iff A*A' = I (partial isometry))
    if( parvalues{8} ) % scale r.h.s. ?
	parvalues{12} = A'*(b./norm(b));
    else
        parvalues{12} = A'*b;
    end
        
    % adapt CGmdoe:
    if( (~sparseA) && (~parvalues{11}) )        % dense, CGmode=0 ==> use linsolve
        opts.SYM = true; opts.POSDEF = true;     
        parvalues{11} = {parvalues{11},A*A',opts};  
        % clear opts;
    else
        if( ~parvalues{11} ) % CGmode=0, but sparse A
            parvalues{11} = {parvalues{11},A*A'};
        %else  % CGmode=1 (dense or sparse)
        %    % nothing to do
        end            
    end
end
[dLBmode,dspl,maxitPr,p,stag,time,tolPr,scale,reset,sparseA,CGmode,x0,useHOC] = deal(parvalues{:});
% clear parvalues parnames nnzA numelA numeldivnnz nnzdivnumel;
