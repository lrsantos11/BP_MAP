function x = CG(A,AAT,b,tol,maxit,CGmode,x0)
%function x = CG(A,AAT,b,tol,maxit,CGmode,x0)
% =========================================================================
%  FILE:           CG.m
%  AUTHOR:         Andreas M. Tillmann, TU Darmstadt, Germany
%  LAST MODIFIED:  09/28/2013
% =========================================================================
%  DESCRIPTION:    A (basic) CG implementation without error checks or
%                  preconditioning, specialized to the solution of normal
%                  equations of the form A*A'*x=b.
%                 
%                  NOTE: The user has to supply consistent data; this
%                        function does not include any error checks!
%
%  INPUT:   A      The matrix yielding the CG system matrix A*A',
%                  can be empty (A=[]) if CGmode=0
%           AAT    The matrix A*A' given explicitly,
%                  can be empty (AAT=[]) if CGmode=1
%           b      The right hand side vector
%           tol    The CG convergence tolerance
%           maxit  The maximal number of CG iterations allowed
%           CGmode Boolean variable to toggle how CG operates:
%                     CGmode = true /1 - only A is used (AAT not needed)
%                     CGmode = false/0 - only AAT is used (A not needed)
%           x0     The starting point (initial solution guess)
%                  [optional; default is the all-zero vector]
%
%  OUTPUT:  x      The (approximate) solution computed by the CG method
% =========================================================================
if( nargin < 7 )
    x = zeros(size(b)); % initial guess: all-zero vector
else
    x = x0;             % initial guess: used-provided
end

if( CGmode ) % A is given (use this for sparse A in particular)
    wtemp = A'*x;    % w = A*wtemp; % i.e., w = A*A'*x      
    r = b - A*wtemp; % r = b - w;
    p = r;
    d = r'*p;       
    for iter = 1:maxit
        if( norm(p,2) < tol ) % convergence w.r.t. accuracy tolerance
            break;            % (small change in x)
        end
        wtemp = A'*p;
        w = A*wtemp;
        alpha = d/(p'*w);
        x = x + alpha*p;
        r = r - alpha*w;
        w = r;
        if( (norm(w,2) < tol) && (norm(r,2) < tol) )
            break;            % convergence (small residual)
        end
        dlast = d;
        d = r'*w;
        p = w + (d/dlast)*p;
    end
    %clear wtemp;

else % AAT=A*A' is given explicitly
        r = b - AAT*x;   % w = AAT*x; % r = b - w;
        p = r;
        d = r'*p;       
        for iter = 1:maxit
            if( norm(p,2) < tol ) % convergence w.r.t. accuracy tolerance
                break;            % (small change in x)
            end
            w = AAT*p;
            alpha = d/(p'*w);
            x = x + alpha*p;
            r = r - alpha*w;
            w = r;
            if( (norm(w,2) < tol) && (norm(r,2) < tol) )
                break;            % convergence (small residual)
            end
            dlast = d;
            d = r'*w;
            p = w + (d/dlast)*p;
        end
end
%clear w r p d iter alpha dlast;
