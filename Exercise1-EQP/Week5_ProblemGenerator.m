function [H, g, A, b] = Week5_ProblemGenerator(n,u_bar,d_0)
arguments
    n 
    u_bar = 0.2
    d_0 = 1
end
    if n >= 3
        H = eye(n+1);
        
        b = [-d_0 zeros(1, n-2) 0]';
        
        g = -u_bar*eye(1,n+1)';
    
        temp = [];
        for i = 1:n-2
            temp = [temp; [zeros(1,i-1) 1 -1 zeros(1,n-i)]];
        end
        
        A = [-1 zeros(1,n-2) 1 0;
             temp;
             zeros(1,n-2) 1 -1 -1]';
    end
end