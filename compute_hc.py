import numpy as np
from constants import *
from ef import solve_ecc, k
from functions import g, F, fp, gT

xy=[-2,0,2]

def f_limit(Mc,eta,K,e):
	M=Mc/(eta**(3/5))
	freq=(K*(1-e**2)/3)**(3/2)*c**3/(2*np.pi*G*M)
	return freq







def log_hc_kshifted(Mc,eta,n_har,n_pts,fi,f0,e0,z):
	M=Mc/(eta**(3/5))
	sum1,freq=[],[]
	f_arr=np.logspace(np.log10(fi),np.log10(f_limit(Mc,eta,0.5,e0)),n_pts)

	itr=4
	count=0

	# fpq=np.zeros((n_pts,n_har,len(xy),itr))
	# kpq=np.zeros((n_pts,n_har,len(xy),itr))
	# epq=np.zeros((n_pts,n_har,len(xy)))
	k_arr=[]
	
	for n,f in enumerate(f_arr):
		s1=0
		s=np.zeros(n_har+1)

		fr=f*(1+z)
		K=k(fr,e0,f0,M,z)
		
		if K>=0.25:
			break

		for p in range(1,n_har+1):
			
			fp=fr/p
			kp=k(fr/p,e0,f0,M,z)
			phi_nq=0
			kpq=np.zeros(3)
			fpq=np.zeros(3)
			epq=np.zeros(3)
			for d,q in enumerate(xy):
				for i in range(itr):
					if kp>=0.5:   #Otherwise fpq becomes negative 
						break

					fpq[q]=fr/(p+q*kp)
					kpq[q]=k(fpq[q],e0,f0,M,z)
					kp=kpq[q]
				if fpq[q]==0:
					break
				epq[q]=	solve_ecc(f0*(1.0+z), e0, fpq[q])
				# if epq[q]==0:
				# 	break
				phi_nq+=g(p,epq[q])[d]/F(epq[q])*1/p**(2/3)
			fac=4*G/(np.pi*c**2*f)*1/3*Mc*c**2/fr*(2*np.pi*G*Mc*fr/c**3)**(2/3)
			s[p]=Mpc**(-3)*fac*phi_nq
			s1+=s[p]

		sum1.append(s1)
		freq.append(f)
				
	
	return np.log10(freq), 0.5*np.log10(sum1), freq, np.sqrt(sum1)



def log_hc_Q(Mc,eta,n_har,n_pts,fi,ff,f0,e0,z):
	M=Mc/(eta**(3/5))
	sum1,freq=[],[]
	f1=np.logspace(np.log10(fi),np.log10(ff),n_pts)
	for f in f1:
		s1=0
		s=np.zeros(n_har+1)
		fr=f*(1+z)
		for n in range(1,n_har+1):
			phi_n=0
			K=k(fr/n,e0,f0,M,z)
			f_n=fr/n
			e_n=solve_ecc(f0*(1.0+z), e0, f_n)
			phi_n+=gT(n,e_n)/(F(e_n)*n**(2/3))
			fac=2/(3*np.pi**2)*(c/f)**3*(2*np.pi*G*Mc*f/c**3)**(5/3)*Mpc**(-3)/((1+z)**(1/3))
			s[n]=fac*phi_n
			s1+=s[n]

		sum1.append(s1)
		freq.append(f)
	return np.log10(freq), 0.5*np.log10(sum1), freq, np.sqrt(sum1)




def hc_kshifted(Mc,eta,n_har,n_pts,fi,f0,e0,z):
	M=Mc/(eta**(3/5))
	sum1,freq=[],[]
	f_arr=np.logspace(np.log10(fi),np.log10(f_limit(Mc,eta,0.5,e0)),n_pts)

	itr=4
	count=0

	# fpq=np.zeros((n_pts,n_har,len(xy),itr))
	# kpq=np.zeros((n_pts,n_har,len(xy),itr))
	# epq=np.zeros((n_pts,n_har,len(xy)))
	k_arr=[]
	
	for n,f in enumerate(f_arr):
		s1=0
		s=np.zeros(n_har+1)

		fr=f*(1+z)
		K=k(fr,e0,f0,M,z)
		
		if K>=0.25:
			break

		for p in range(1,n_har+1):
			
			fp=fr/p
			kp=k(fr/p,e0,f0,M,z)
			phi_nq=0
			kpq=np.zeros(3)
			fpq=np.zeros(3)
			epq=np.zeros(3)
			for d,q in enumerate(xy):
				for i in range(itr):
					if kp>=0.5:   #Otherwise fpq becomes negative 
						break

					fpq[q]=fr/(p+q*kp)
					kpq[q]=k(fpq[q],e0,f0,M,z)
					kp=kpq[q]
				if fpq[q]==0:
					break
				epq[q]=	solve_ecc(f0*(1.0+z), e0, fpq[q])
				# if epq[q]==0:
				# 	break
				phi_nq+=g(p,epq[q])[d]/F(epq[q])*1/p**(2/3)
			fac=4*G/(np.pi*c**2*f)*1/3*Mc*c**2/fr*(2*np.pi*G*Mc*fr/c**3)**(2/3)
			s[p]=fac*phi_nq
			s1+=s[p]

		sum1.append(s1)
		freq.append(f)
				
	
	return  freq, sum1


def hc_Q(Mc,eta,n_har,n_pts,fi,ff,f0,e0,z):
	M=Mc/(eta**(3/5))
	sum1,freq=[],[]
	f1=np.logspace(np.log10(fi),np.log10(ff),n_pts)
	for f in f1:
		s1=0
		s=np.zeros(n_har+1)
		fr=f*(1+z)
		for n in range(1,n_har+1):
			phi_n=0
			K=k(fr/n,e0,f0,M,z)
			f_n=fr/n
			e_n=solve_ecc(f0*(1.0+z), e0, f_n)
			phi_n+=gT(n,e_n)/(F(e_n)*n**(2/3))
			fac=4*G/(np.pi*c**2*f)*1/3*Mc*c**2/fr*(2*np.pi*G*Mc*fr/c**3)**(2/3)
			s[n]=fac*phi_n
			s1+=s[n]

		sum1.append(s1)
		freq.append(f)
	return freq, np.array(sum1)