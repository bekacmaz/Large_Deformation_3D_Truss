from math import log,sqrt,exp
import numpy as np

def get_FINT_NL(MG, numel, DOF, NC, x, X, eps_pl_n, eps_pl_bar, lambda_p_n):
    dim = int(DOF.max())
    fint = np.zeros( (dim,1) )
    eps_pl_n_temp_func = np.zeros( (dim,1) )
    eps_pl_bar_temp_func = np.zeros( (dim,1) )
    lambda_p_temp_func = np.zeros( (dim,1) )
    KK = np.zeros( (dim, dim) )
    
    for i in range(numel):
        bn_id = int( NC[i,1]-1 )
        en_id = int( NC[i,2]-1 )
        dist_vec = x[en_id,1:]-x[bn_id,1:]
        Dist_vec = X[en_id,1:]-X[bn_id,1:]
        l = (np.matmul(dist_vec, np.transpose(dist_vec)))**0.50
        L = (np.matmul(Dist_vec, np.transpose(Dist_vec)))**0.50
        
        el_eps_pl_n = eps_pl_n[i,0]
        el_eps_pl_bar_n = eps_pl_bar[i,0]
        el_lambda_pl_n = lambda_p_n[i,0]
        
        el_l_pl_n = el_lambda_pl_n*L
        
        vol_zero = L * MG[i,0]
        area_current = vol_zero / l
        vol_current = area_current * l
    
        eps_new = log(l/L)
        eps_e_new_tr = eps_new - el_eps_pl_n
        tau_new_tr = eps_e_new_tr*MG[i,1]
        f_new_tr = abs(tau_new_tr)-(MG[i,3]+(MG[i,2]*el_eps_pl_bar_n))
        
        E = MG[i,1]
        H = MG[i,2]
        
        if f_new_tr < 0:
            tau_new = tau_new_tr
            el_eps_pl_new = el_eps_pl_n
            el_eps_pl_bar_new = el_eps_pl_bar_n
            el_lambda_new = el_lambda_pl_n
            e = E
            
        if f_new_tr >= 0:
            del_gama = (f_new_tr)/(E+H)
            del_eps_pl = del_gama * np.sign(tau_new_tr)
            el_eps_pl_new = el_eps_pl_n + del_eps_pl
            tau_new = tau_new_tr - (E * del_eps_pl)
            el_lambda_new = (el_l_pl_n*exp(del_eps_pl))/L
            el_eps_pl_bar_new = el_eps_pl_bar_n + del_gama
            e = ((E * H)/(E + H))
            
        T = (vol_zero/vol_current)*area_current * tau_new     
        n = dist_vec/l
        k = vol_zero/(l**2) * (e - (2*tau_new))
        
        n_dyadic_n = [ [n[0]* n[0], n[0]* n[1], n[0]* n[2]],
        [n[1]* n[0], n[1]* n[1], n[1]* n[2]],
        [n[2]* n[0], n[2]* n[1], n[2]* n[2]] ]
        
        Kaa = np.multiply(k, n_dyadic_n) + np.multiply(T/l, np.eye( 3 ))
        Kbb = Kaa
        Kba = -Kbb
        Kab = Kba
        
        K1 = np.hstack((Kaa, Kab))
        K2 = np.hstack((Kba, Kbb))
        
        K = np.vstack((K1, K2))
        
        stress = tau_new
        Tb = stress * area_current * n
        Ta = -Tb
        T_global = np.concatenate((Ta,Tb), axis=0)
        
        eldof = [ int(DOF[NC[i,1]-1,0]), int(DOF[NC[i,1]-1,1]), int(DOF[NC[i,1]-1,2]), \
                 int(DOF[NC[i,2]-1,0]), int(DOF[NC[i,2]-1,1]), int(DOF[NC[i,2]-1,2]) ]
        
        for ii in range(6):
            for jj in range(6):
                KK[eldof[ii]-1,eldof[jj]-1]+=K[ii,jj]   
        
        for ii in range(6):
            fint[eldof[ii]-1,0]+= T_global[ii]
            
        eps_pl_n_temp_func[i,0] = el_eps_pl_new
        eps_pl_bar_temp_func[i,0] = el_eps_pl_bar_new
        lambda_p_temp_func[i,0] = el_lambda_new
        
    return KK, fint, eps_pl_n_temp_func, eps_pl_bar_temp_func, lambda_p_temp_func