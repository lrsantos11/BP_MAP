function [x,fval,err,exfl,it] = ISAL1bpdn(A,b,delta,varargin)
%function [x,fval,err,exfl,it] = ISAL1bpdn(A,b,delta,varargin)
% =========================================================================
%  FILE:           ISAL1bpdn.m
%  VERSION:        1.00 (v1.0)  --- prototype ---
%  AUTHOR:         Andreas M. Tillmann, TU Darmstadt, Germany
%  LAST MODIFIED:  09/28/2013
% =========================================================================
%  DESCRIPTION:    This is ISAL1bpdn, an Infeasible-Point Subgradient 
%                  Algorithm for the L1-norm minimization problem
%                       (P1dn):  min ||x||_1   s.t. ||Ax-b||_2 <= delta 
%                  (aka Basis Pursuit Denoising)
%
%                  References:
%                  * Lorenz/Pfetsch/Tillmann, "An Infeasible-Point Subgradient
%                  Method Using Adaptive Approximate Projections", 
%                  Computational Optimization and Applications, 2013 (to
%                  appear), DOI: 10.1007/s10589-013-9602-3
%                  * Lorenz/Pfetsch/Tillmann, "Solving Basis Pursuit: Heuristic
%                  Optimality Check and Solver Comparison", 2011/12, preprint,
%                  http://www.optimization-online.org/DB_HTML/2011/07/3100.html
%                  * Tillmann, "Computational Aspects of Compressed Sensing", 
%                  submitted dissertation, 2013
%
%                  Copyright (C) Andreas M. Tillmann, 2013
%
%                  *** projections are realized using code from UNLocBoX,
%                  see proj_bpdnCons.m for details and authorship info.
%
%  INPUT:  ------- REQUIRED: 
%          A       Coefficient matrix of the linear constraint system
%          b       Right hand side vector of the system
%          delta   noise estimate / allowed deviation from equality Ax=b
%
%          ------- OPTIONAL: (must be passed as string-value-pairs)
%          useHOC  Toggle usage of Heuristic Optimality Check (HOC)
%                  (see also 'freq' in ISAL1bpdn_Initialize function below);
%                  [ default is useHOC = true/1 ]
%          dLBmode Specifies the (constant) dual lower bound dLB used in  
%                  the step size function and the heuristic method used to 
%                  compute it. Feas. values for dLBmode range from -1 to 6; 
%                  see ISAL1bpdn_lowbo.m for a description. 
%                  For dLBmode=-1, dLB is calculated using HOC (which is 
%                  then forced "on", possibly overriding useHOC setting),  
%                  and no iterative updates are applied during the algo.;
%                  [ default is dLBmode = -1 ]
%          dspl    Controls the amount of displayed output, i.e., every
%                  dspl iterations an overview of the current progress is
%                  printed onscreen. Passing the value dspl=inf will sup-
%                  press all such output;
%                  [ default is dspl=inf ]
%          time    Limit on the running time of the procedure (in seconds); 
%                  [ default is time=300 (5 minutes) ]
%          p       Defines the number of iterations with no or no relevant
%                  (i.e.,<1e-6*f(xk)) improvement of the current best ob-
%                  jective value after which the the constant factor in
%                  the step size function is halved. A maximum number of 
%                  iterations allowed to use the same value of p is defined
%                  in the program (parameter 'maxitcurrp'), in order to 
%                  avoid long periods with few (but relevant) improvements
%                  in early stages of the iterative process;
%                  [ default is set in ISAL1_setparams.m ] 
%          stag    Number of iterations after which the algorithm stops
%                  (or initiates a restart, see 'reset' parameter);
%                  [ default is stag = 500 ]                  
%          tolPr   initial convergence tolerance for approximate projection (in [0,1]);
%                  [ default is tolPr = 0.85 ]
%          maxitPr Initial maximum number of iterations for approx. projection;
%                  [ default is maxitPr = 20 ]
%          scale   Indicator for scaling of right hand side vector to have
%                  unit Euclidean norm (this improves numerical robust-
%                  ness of the method);
%                  [ default is scale = 1 ]
%          reset   Indicator for automatic reset / restart of the method
%                  (which stricter algorithmic parameter settings) in case
%                  stagnation was detected or the step sizes became too 
%                  small before an optimal solution was reached.
%                  [ default is reset = 1 ]
%          CGmode  Indicator for the CG implementation to be used (in HOC), CGmode=1      
%                  corresponds to solving normal equations with implicit or
%                  "split" computation of products A*A'*x, while CGmode=0
%                  assumes A*A' is given explicitly.
%                  [ default depends on whether the number of nonzero ele-
%                    ments of A is smaller than 2 times the squared row
%                    number of A (in this case, split computation should
%                    be faster than working with (possibly full) A*A'),
%                    see ISAL1bpdn_setparams.m ]
%          x0      Starting point for the algorithm, should be (at least
%                  very close to) feasible to ensure its L1-norm is larger 
%                  than the lower bound; 
%                  for a description of possible predefined choices for x0,      % Note: x0-choices have not been adapted to BPDN yet w.r.t feasibility
%                  see ISAL1bpdn_setparams.m;
%                  [ default is A'*b (if scale=1, A'*b./norm(b,2)),
%                    theoretical possibility of ||x0||_1 < dLB ignored ]
%
% OUTPUT:  x       The (approximately) feasible point with the best objec-
%                  tive function value; if exfl=1, x is an (approximate)
%                  solution to the L1-norm minimization problem (P1dn)
%          fval    The L1-norm of the returned solution x
%          err     A measure for (primal) feasibility: max{||Ax-b||_2-delta,0}
%          exfl    Exitflag, indicating reason for program termination:
%                   0 =  nothing happened (something went wrong)
%                   1 =  provably optimal solution reached (HOC success)
%                   2 =  stagnation of progress w.r.t. objective decrease
%                   4 =  time limit exceeded
%                   8 =  step size became smaller than eps (~2.2204e-16)
%                  16 =  stagnation of approximate support
%                  32 =  did not move away from starting point (i.e., no
%                        approx. feasible point w/ better obj. was found)
%                  - A negative sign indicates a restart was performed.
%                  In all cases the returned x is the best (approximately)
%                  feasible point (not necessarily optimal) reached so far.
%          it      Number of iterations performed
% =========================================================================
t   = tic;  % stop total program execution time
exfl = 0;   % "nothing happened"
term_msg = 'Nothing happend / Something went wrong.';

% Quick return:
if( isempty(A) || isempty(b) || size(A,1)~=size(b,1) ) % no true instance
    [x,fval,err,it] = deal([]); 
    return;
elseif( delta > norm(b)-1e-12 ) % the solution is the all-zero vector
    x = zeros(size(A,2),1); exfl = 1; [fval,err,it] = deal(0);
    return;
end

% Initialize parameters and variables :
[dLBmode,dspl,maxitPr,p,stag,time,tolPr,m,n,scale,reset,sparseA,CGmode,x,useHOC] = ISAL1bpdn_setparams(A,b,delta,varargin{:});
AAT=[];opts=[];
ISAL1bpdn_Initialize; % (nested function, see below)

% Iteration information display header:
if( ~isinf(dspl) )
    fprintf('\n  iter  |     ||x||_1     |      fbest      |    step size    | ||Ax-b||-delta  | ~supp(x) |  est. duality gap\n%s\n',...
        '-------------------------------------------------------------------------------------------------------------------');
end

% ==================== MAIN LOOP START ====================================
while( (~opt) && (imp < stag) && (alphak > eps) && (toc(t) <= time) && (stagextend < stag) && (suppstag < suppstaglimit) )   
    % Display iteration information, if applicable:
    if( mod(it,dspl) == 0 )
        feascurr = max( norm(A*xk-b), 0);
        fprintf('%6d  |  %1.7e  |  %1.7e  |  %1.7e  |  %1.7e  |  %6d  |  %1.7e %%\n',it,fcurr,fval,alphak,feascurr,suppxk,100*(fval-dLBbest)/fval);   
    end
    % Increment counters:
    it = it + 1; imp = imp + 1; currp = currp + 1;
    
    if( (~reset) && (~praised) && (imp>0.75*stag) )  % if after a reset stagnation seems
        p = round(1.5*p);                            % likely, increase p and stag again
        stag=round(1.5*stag);                        % in the hope of improvment
        praised = 1; % avoid doing this here more than once
    end
    
    % Compute next (approximately) projected iterate:
    xk = proj_bpdnCons(xk-alphak*hk,@(v)A*v,@(v)A'*v,b,delta,normA, useFISTA, tolPr, maxitPr, tight);
    
    % update constant factor in step size function, if necessary:
    if( ( mod(imp+stagextend,p) == 0 ) || ( currp == maxitcurrp ) )        
        pfact = pfact/2;    currp = 0;
        
        maxitPr = min(maxitPr+5,100); tolPr = max(pfact,1e-5); % increase proj. accuracy --- Note: currently tied to pfact        
    end
    
    % update function values, best iterate, feasibility measure etc:
    fcurr = norm(xk,1); 
    if( fcurr < fvalinfeas )    % improvement in objective function value
        if( fvalinfeas-fcurr >= (1e-6)*fvalinfeas ) % relevant improvement
            imp = 0; stagextend = 0;
        end
        fvalinfeas = fcurr;
        feascurr = max( norm(A*xk-b)-delta, 0 );             
        if( feascurr < 1e-6 )   % feasibility, with respect to tolerance   
            fval = fcurr; x = xk; err = feascurr;
        end
    end
    
    % update subgradient:
    H{mod(it,4)+1} = sign(xk); % (unstabilized) subgradient of xk
    if( dLBmode == 5 )
        hkold = hk;            % not needed otherwise
    end
    % new (stabilized) subgradient of xk:
    hk = 0.6*H{mod(it,4)+1} + 0.2*H{mod(it-1,4)+1} + 0.1*(H{mod(it-2,4)+1} + H{mod(it-3,4)+1});
    
    % reset point to current best if points "go astray" by too much        
    if( fcurr > 1e3*fval )
        xk = x;
    end
    % this sometimes allows to overcome large deviations at the beginning
    % (only xk is reset; the aggregated subgradient information of
    % iterations up to the current iteration is used in the next
    % iteration), a problem much alleviated by scaling of the r.h.s. and
    % choosing initial pfact-values below 1.
    % May be fruitless if the deviations are due to divergence of the
    % distance from iterate points to the optimal solution set; this will
    % nevertheless be "detected" as stagnation without becoming feasible.
    
    % Perform extra "artificial" iterations if stagnation occurs but no
    % feasible point has been found so far (or none better than x0):
    % (if reset is switched on, this is only done after the reset)
    if( ((~reset) && (imp == stag)) && ( (err > 1e-6) || (fval == fval0) ) )
        imp = imp - 1;           % avoid termination due to "imp == stag"
        stagextend = stagextend + 1; % increment artificial it. counter
    end
    
    % If the current iterate's subgradient differs from the last iterate's,
    % or if a better objective value has been achieved with the current 
    % (possibly infeasible) iterate, try to improve the dual lower bound,
    % if updates are switched on (by default, they are not, because 
    % they currently only serve to update the iteration-log display):
    if( ( dLBmode >= 0 ) && (( ~all(H{mod(it,4)+1}==H{mod(it-1,4)+1}) ) || ( imp == 0 )) )
        if( dLBmode == 5 )
 	    dLBnew = ISAL1bpdn_lowbo(A,b,xk,delta,hkold,dLBmode);
        else
	    dLBnew = ISAL1bpdn_lowbo(A,b,xk,delta,hk,dLBmode);
        end
        if( dLBnew > dLBbest )  % keep best lower bound found so far
            dLBbest = dLBnew;            
        end
    end
    
    % Support Approximation & HOC (performed only every 'freq' iterations, or 
    % if termination due to too-small step sizes is imminent):
    if( (mod(it,freq)==0) || (alphak<eps) )
        xsort = sort(abs(xk(abs(xk)>minsupptol)),'descend');       % Dynamic support approximation: take those indices
        s = length(xsort);                                         % that together account for given percentage of L1-norm
        if( s > 0 ) % might not be the case if all entries in x are too small ...
            temp = 2; supptol = xsort(1); 
            energy = percentage*fcurr;                             % (percentage=0.9999 set by default in ISAL1_Initialize below)  
            while( ( supptol < energy ) && (temp <= s) )
                supptol = supptol + xsort(temp);
                temp = temp + 1;
            end
            if( temp == s+1 )
                supptol = minsupptol;
            else
                supptol = max(xsort(temp-1),minsupptol);
            end
            clear xsort;
            Suppprev = Suppxk; suppprev = suppxk; 
            [Suppxk,suppxk] = supp(xk,supptol); 
            suppsame = ((suppprev==suppxk) && all(Suppxk==Suppprev)); 
            if( (useHOC) && (suppxk <= m) && (~suppsame) )         % only try HOC if support has changed and its size is at most m
                ISAL1bpdn_HOC;
            end
            if( suppsame )
                suppstag = suppstag+1;
            else
                suppstag = 0;
            end
            clear Suppprev suppprev;
        end
    end
    
    % update the stepsize:
    alphak = pfact*(fcurr-dLB)/(norm(hk,2)^2);
    
    % ===== BEGIN RESET (if switched on) ==================================
    if( ( (alphak<eps) || (imp==stag) || (suppstag==suppstaglimit) ) && (reset) && (~opt) )
        reset = false;
        maxitPr=5; % tolPr = 0.85; % resetting tolPr apparently doesn't change much in the present configuration ...
        imp = 0; stagextend = 0; suppstag = 0;
        p=5*p; stag = max(10000,10*p); 
        alphak = 0.85*alphak/pfact; pfact = 0.85; 
        % alphak = pfact*(fcurr-dLB)/(norm(hk,2)^2); 
        maxitcurrp = max( round(stag/sqrt(2)) , round(stag/(1+nnz(A)/numel(A))^2) );                     
        % additional optimality test, if not already done in this iter.:
        if( (mod(it,freq)~= 0) && (imp==stag) )
            xsort = sort(abs(xk(abs(xk)>minsupptol)),'descend'); 
            s = length(xsort); 
            if( s > 0 ) % might not be the case if all entries in x are too small ...
                temp = 2; supptol = xsort(1); energy = percentage*fcurr;
                while( ( supptol < energy ) && (temp <= s) )
                    supptol = supptol + xsort(temp);
                    temp = temp + 1;
                end
                if( temp == s+1 )
                    supptol = minsupptol;
                else
                    supptol = max(xsort(temp-1),minsupptol);
                end
                clear xsort;
                Suppprev = Suppxk; suppprev = suppxk;
                [Suppxk,suppxk] = supp(xk,supptol);
                suppsame = ((suppprev==suppxk) && all(Suppxk==Suppprev));
                if( (useHOC) && (suppxk <= m) && (~suppsame) )   
                    ISAL1bpdn_HOC;                 
                end
                if( suppsame )
                    suppstag = suppstag+1;
                else
                    suppstag = 0;
                end
                clear Suppprev suppprev;
            end
        end
    end
    % ===== END OF RESET===================================================
    
end
% ==================== END OF MAIN LOOP ===================================

% un-scale the current best solution, if necessary, 
% and try to obtain "sparsified" solution from current approx. support
% (essentially a debiasing step, but with the following restriction):
% If this yields a feasible sol. with smaller L1-norm than former best x,
% the better solution is taken instead.

% seek feasible solution on approx. support of last iterate (after unscaling...): 
[Suppxk,suppxk] = supp(x,supptol);     
if( (scale) ) % && (~opt) imlied, since if opt=1, scale is set to false/0
    % unscale current best solution:          
    xk = x;
    x = x.*normborig; fval = norm(x,1); err = max( norm(A*x-borig)-deltaorig, 0);
    b = borig; delta = deltaorig; 
    dLBbest = normborig*dLBbest;
    scale = 0;
end 

if( ~opt )
    if( ~isinf(dspl) )
        fprintf('try improvement / debiasing\n');
    end
    ASx = A(:,Suppxk);
    normASx = normest(ASx)^2;%svds(ASx,1)^2; %=norm(ASx)^2; % could also just use normA, but better bound is better for proj. algo.
    xtest = spalloc(n,1,suppxk);
    xtest(Suppxk) = proj_bpdnCons(x(Suppxk),@(v)ASx*v,@(v)ASx'*v,b,delta,normASx,useFISTA, 1e-6, 200, tight);  % if A tight then also ASx (?)
    temp = norm(xtest,1);
    s = max( norm(A*xtest-b)-delta, 0);
    if( (temp < fval) && (s < 1e-6) ) % obtained new solution is current best
        err = s; fval = temp; x = xtest; xk = xtest;      
    else % try to recover from possible too-strict support approximation setup       
        [Suppxk,suppxk] = supp(x,supptol/10);
        ASx = A(:,Suppxk);
        normASx = normest(ASx)^2;%svds(ASx,1)^2; %norm(ASx)^2; % could also just use normA, but better bound is better for proj. algo.
        xtest = spalloc(n,1,suppxk);
        xtest(Suppxk) = proj_bpdnCons(x(Suppxk),@(v)ASx*v,@(v)ASx'*v,b,delta,normASx,useFISTA, 1e-6, 200, tight);  % if A tight then also ASx (?)
        temp = norm(xtest,1);
        s = max( norm(A*xtest-b) - delta, 0);
        if( (temp < fval) && (s < 1e-6) ) % obtained new solution is current best
            err = s; fval = temp; x = xtest; xk = xtest;   
            supptol = 1e-6;                 
        end        
    end
    
    %ISAL1bpdn_HOC;
    
    if err > 1e-6 % unsatisfactory accuracy w.r.t. feasibility         
        x = proj_bpdnCons(x, @(v)A*v, @(v)A'*v, b, delta, normA, useFISTA, 1e-8, 200, tight);
        fval = norm(x,1);
        err = max( norm(A*x-b) - delta, 0);
    end    

else % try to improve solution if support was overestimated and 
     % HOC (perhaps falsely) declared success due to numerical issues
     % note: here, we are already unscaled again.
     [Suppxk,suppxk] = supp(x,1e-6*norm(b));
     ASx = A(:,Suppxk);
     normASx = normest(ASx)^2;%svds(ASx,1)^2; %norm(ASx)^2; % could also just use normA, but better bound is better for proj. algo.
     xtest = spalloc(n,1,suppxk);
     xtest(Suppxk) = proj_bpdnCons(x(Suppxk),@(v)ASx*v,@(v)ASx'*v,b,delta,normASx,useFISTA, 1e-9, 300, tight);  % if A tight then also ASx (?)
     temp = norm(xtest,1);
     s = max( norm(A*xtest-b) - delta, 0);
     if( (temp < fval) && (s < 1e-6) ) % obtained new solution is current best
         err = s; fval = temp; x = xtest; xk = xtest;
         supptol = 1e-9*normborig;   
     end
     ISAL1bpdn_HOC;   
end
            
% Set exitflag:
if( opt == 1 )
    exfl = round(-0.5+reset);
    term_msg = 'HOC Success - Optimal solution recovered.';
elseif( imp == stag )
    exfl = round(-0.5+reset)*2;
    term_msg = 'Stagnation of algorithmic progress (insufficient improvement).';
elseif( toc(t) > time )
    exfl = round(-0.5+reset)*4;
    term_msg = sprintf('Time limit (%.2f sec) exceeded.',time);
elseif( alphak <= eps )
    exfl = round(-0.5+reset)*8;
    term_msg = 'Stepsize became too small.';
elseif( suppstag == suppstaglimit ) 
    exfl = round(-0.5+reset)*16;
    term_msg = 'Stagnation of algorithmic progress (invariant approx. support).';
elseif( fval == fval0 ) 
    exfl = round(-0.5+reset)*32;
    term_msg = 'No feasible point with better objective than starting point was found.';
end

% Display solution information (if applicable):
if( ~isinf(dspl) )
    fprintf('-------------------------------------------------------------------------------------------------------------------\n');
    fprintf('%6d  |  %1.7e  |  %1.7e  |  %1.7e  |  %1.7e  |  %6d  |  %1.7e %%\n',it,fval,fval,alphak,err,numel(supp(x,supptol)),100*(fval-dLBbest)/fval);   
    fprintf('-------------------------------------------------------------------------------------------------------------------\n');
    fprintf('  Termination cause: %s\n',term_msg);
    fprintf('  Best dual obj. val. found: %1.7e (==> Duality Gap Estimate: %2.3f %%)\n',dLBbest,100*(fval-dLBbest)/fval);
    fprintf('  Total time elapsed: %.2f sec\n',toc(t));    
end

% ==================================================================================================
%% Nested functions:
% ==================================================================================================
    function ISAL1bpdn_Initialize
        % This function sets up several tolerance values, parameters, counters and initial values 
        % of variables (iterate points, stepsizes, subgradients etc).       
        
        % scale right hand side, if scaling is turned on:
        borig = b; 
        normborig = norm(borig); % Euclidean norm of original r.h.s.
        deltaorig = delta;
        if( scale )
            b = b./normborig;
            delta = delta/normborig;
        end
        
        AAT = []; opts = [];
        %sparseA = issparse(A);    % true iff A is stored in sparse format
        % if CGmode = false/0 after initialization, it is a cell variable :
        if( iscell(CGmode) )      % then A*A' has already been computed, ...
            AAT = CGmode{2};
            if( ~sparseA )        % ... and so has opts, if A is stored in dense format
                opts = CGmode{3}; % opts contains s.p.d. options for linsolve 
            end
            clear CGmode;
            CGmode = false;
        else  % CGmode = 1
            if( (~sparseA) && (nnz(A)<0.1*m^2) ) % the matrix is relatively sparse,
                A = sparse(A);                   % storage in sparse format will
                sparseA = 1;                     % likely speed up computations
            end
        end
        
        % Initializate variables and counters:
        fval     = norm(x,1);  % currently best function value (L1-norm of x0)
        % if x0 = 0 (and b ~= 0), step size is undefined; in this case, override x0 by all-ones vector:
        if( fval < eps ), x = ones(size(x)); fval = n; end
        xk       = x;          % current iterate
        fval0    = fval;       % objective function value of starting point
        fcurr    = fval;       % current function value (L1-norm of xk)          
        fvalinfeas = fval;     % current best objective value of infeasible points
        err = max( norm(A*x-b)-delta, 0); % measure for feasibility of best "feasible" point
        pfact    = 0.85;       % initial factor in step size function (alpha) 
        maxitcurrp = round( max( stag/(1+nnz(A)/numel(A))^2 , stag/sqrt(2) ) ); 
                               % max. number of iter's w/o reducing pfact
        %lambda = zeros(m,1);   % use sol. from prev. iter. as CG starting point
        freq = fix(m/100);     % max. frequency of optimality tests (HOC)
        it       = 0;          % iteration counter
        imp      = 0;          % counts iterations without relevant improvement
        stagextend = 0;        % Used to extend allowed subsequent iterations with-
                               % out improvement, e.g. to give the algorithm a bet-
                               % ter chance to reach feasibility or leave the star-
                               % ting point
        currp    = 0;          % number of subseq. iterations with the same pfact                       
        praised  = 0;          % if, after a restart, stagnation seems likely, p is
                               % increased once more and praised set to 1
        minsupptol = 1e-6;     % minimal threshold for support approximation
        percentage = 0.9999;   % percentage of obj. val. approx. supp. should carry
        suppstag = 0;          % counter for support stagnation
        suppstaglimit = 10;    % terminate if same support in 10*freq it's
        useFISTA = true;       % method used for approximate projection (FISTA by default, else ISTA)
        normA = normest(A)^2;  % need a (good) estimate of norm(A,2)^2 for projection method
                               % exact choices svds(A,1)^2 or norm(A)^2 take way too much time; normest also depends on size of A but scales a bit better
                               % --- note: in principle, the backtracking-FISTA could be used as well
        tight = false;         % usually, A is not a tight frame
        %if ( max(max(abs(A*A'-normA*eye(size(A,2))))) <= 1e-6 )         % too expensive...
        %   tight = true; % A is (at least very close to) a tight frame
        %end
        % --- note: currently, the tight-frame case is not treated differently from the general case at all ...
        
        % Subgradient choice : sign(x) is one subgradient of ||x||_1 at x
        H = cell(1,4);       % preallocate memory for H                   
        for i = 1:4          % initialize array of last 4 subgradients
            H{i} = sign(x);  % the subgradient of ||x||_1 at x0 used in the ISAL1
        end
        % The stabilized subgradient (replaces use of single subgradients):
        hk = sign(x);        % = 0.6*H{1}+0.2*H{2}+0.1*(H{3}+H{4});
        
        % Compute support approx. threshold (support carries ~99.99% of ||xk||_1): 
        xsort = sort(abs(full(x(abs(x)>minsupptol))),'descend');
        supptol = minsupptol;
        s = numel(xsort); 
        if( s > 0 ) % might not be the case if all entries in x are too small ...
            temp = 2; supptol = xsort(1); energy = percentage*fval;
            while( ( supptol < energy ) && ( temp <= s ) )
                supptol = supptol + xsort(temp);
                temp = temp + 1;
            end
            if(temp == s+1)
                supptol = minsupptol;
            else
                supptol = max(xsort(temp-1),minsupptol);
            end
        end
        [Suppxk,suppxk] = supp(x,supptol);
        clear xsort;
        
        % Compute the constant dual lower bound dLB used in step size:
        dLB = 0;
        dLBbest = dLB;
                
        ISAL1bpdn_HOC;        
        dLBbest = max(dLBbest,dLB); % keep best lower bound obtained so far
        dLB = dLBbest;
        
        % The initial step size including the lower bound dLB:        
        alphak = pfact*(fcurr-dLB)/(norm(hk,2)^2);
        % Prevent starting with too small a step size (may hinder algorithmic progress): 
        if( alphak < pfact/2 )                % <=> fcurr-dLB < norm(hk,2)^2/2;
            dLB = max(fcurr-norm(hk,2)^2,0);  
        end
                
    end % of nested function ISAL1bpdn_Initialize
% ==================================================================================================

% ==================================================================================================
% Heuristic Optimality Check (HOC); with some additional ISAL1bpdn-specific computations:        
    function ISAL1bpdn_HOC    
        % The stabilized subgradient hk is used instead of the sign, as in the rest of ISAL1bpdn.
        % If opt=1 is the result, the following values are modified: x, scale, fval, err, dLBbest
        opt = 0;
        xtest = spalloc(n,1,suppxk);
        
        HOCfeastol = 1e-3; HOCopttol = 1e-6;   % Tolerance parameters; current settings seem ok, but sometimes lead to false-positive success claims
        
        ASx = A(:,Suppxk);  
        % assume that rank(ASx) = suppxk ...
        
        u = ASx'*borig;           
        s = hk(Suppxk); 
        if( ~CGmode )
            ASxTASx = ASx'*ASx;
        else
            ASxTASx = [];
        end
        % compute (approx.) sol. qu to ASx'*ASx*q = u, and
        % compute (approx.) sol. qs to ASx'*ASx*q = s :        
        if( ~sparseA && ~CGmode )                      
            if(issparse(u)), u=full(u); end;
            if(issparse(s)), s=full(s); end;
            if( min(eig(ASxTASx))<=1e-12 )   % ASx is (numerically) not pos. def.
                [qu,~] = linsolve(ASxTASx,u);
                [qs,~] = linsolve(ASxTASx,s);
            else
                [qu,~] = linsolve(ASxTASx,u,opts);
                [qs,~] = linsolve(ASxTASx,s,opts);
            end
        else
            qu = CG(ASx',ASxTASx,u,1e-9,25,CGmode,spalloc(suppxk,1,suppxk));  % reliability of HOC apparently largely depends on CG accuracy here!
            qs = CG(ASx',ASxTASx,s,1e-9,25,CGmode,spalloc(suppxk,1,suppxk));  % may be ok to use 20 iter.s with sign-vector r.h.s. (as in HOC for BP)
        end       
        
        % Lagrange multiplier estimate :
        temp = ASx*qs;
        ls = sqrt( ( norm(temp)^2 ) / (deltaorig^2 - normborig^2 + qu'*u) );
        
        xtest(Suppxk) = qu - (1/ls)*qs;
        
        ys = ls*(ASx*xtest(Suppxk)-borig);
        
        % verify primal and dual feasibility:
        primfeas = ( abs( norm(A*xtest - borig) - deltaorig ) <= HOCfeastol );        
        temp = norm(A'*ys,'inf');
        dualfeas = ( abs( temp - 1 ) <= HOCfeastol );
        
        % dual bound update: (w.r.t. scaled problem, if scaling active)       
        dLBnew = -b'*(ys./temp)-delta*norm(ys);  % always valid
        if( dLBnew > dLBbest )   % keep best lower bound found so far
            dLBbest = dLBnew;    % (overrides former value of dLBbest, no algorithmic consequence)
        end
        
        % check (relative) duality gap:
        temp = norm(xtest(Suppxk),1);
        gap = abs( temp +borig'*ys + deltaorig*norm(ys) ) / temp; % -normborig*dLBnew except when unscaled again...
        if primfeas && dualfeas && ( gap <= HOCopttol )
            opt = 1;
            x = xtest;
            err = max( norm(A*x - borig) - deltaorig, 0 );
            fval = temp;
            if( scale )
                scale = 0; % opt. sol. computed here is w.r.t. unscaled data!
                b = borig; delta = deltaorig;
                dLBbest = dLBbest*normborig;
            end;
        end        
        % clear ASx temp w dLBnew;        
        
    end % of nested function ISAL1bpdn_HOC
% ==================================================================================================

end % of function ISAL1bpdn
