module all
        double precision, allocatable ::  TP(:),prax(:,:,:),strain(:,:),TP_err(:),Vol(:)
        double precision Pc
         integer i_all
end module all

program cgistrain 
	use all
	implicit none
	
	double precision, allocatable:: a(:,:)!lattice parameters in a,b,c,al,be,ga order
	double precision b(3,4) !pressure fit parameters
	double precision BM(4) ! Birch Murnaghan parameters
	
	double precision base(3,3) !base T or P
	double precision avg(3,3) !avg orthonormal matrix
	double precision iS(3,3) !inverse of new conversion  matrix 
	double precision Id(3,3)!Identity
	double precision alpha(3),err_alp(3)!linear dependance
	double precision dep(3),err_dep(3)!linear offset
	double precision, allocatable:: K(:,:)!compressibility
	
	double precision, allocatable:: Jac(:,:)!Jacobian
	double precision, allocatable:: err_k(:,:)
    double precision, allocatable::  covar(:,:),tp_err_spare(:)!covariance matrix
	double precision e(3,3) !total displacement
	double precision ep(3,3) !strain
	double precision cr_ax(3,3) !principal axes,pr_ax in crystal coordinates along axes
	double precision work(192),funct !Dimensions chosen by lapack, apparently
	
 	integer maxfn,nsig,iout,k_max,n_all

	character TorP,ref_pt,pt,finite,euler
	character*44 input
	double precision ga_s,gb_s !gamma star 
	
	double precision pi
	double precision pconv
	integer i,j,u,i2piv(2,2),ipiv(3,3),info,i4piv(4,4)
	integer it,jt,kt,lt
	double precision best(3),it_fac(4),best_b(3,4),temp(3),f0

	double precision cellvol
	
	INTEGER, PARAMETER :: grid = 5
	
	INTERFACE

	
	subroutine press_fit(m,n,b,fvec,info)
		use all
		implicit none
		INTEGER, INTENT(IN)    :: m !no of functions
		INTEGER, INTENT(IN)    :: n !size o'vector	
		double precision, INTENT(IN)  :: b(n)
		double precision, INTENT(out) :: fvec(m)
		integer, intent (out) :: info
	end subroutine

	subroutine birch_murn(m,n,b,fvec,info)
		use all
		implicit none
		INTEGER, INTENT(IN)    :: m !no of functions
		INTEGER, INTENT(IN)    :: n !size o'vector	
		double precision, INTENT(IN)  :: b(n)
		double precision, INTENT(out) :: fvec(m)
		integer, intent (out) :: info
	end subroutine

	subroutine sec_birch_murn(m,n,b,fvec,info)
		use all
		implicit none
		INTEGER, INTENT(IN)    :: m !no of functions
		INTEGER, INTENT(IN)    :: n !size o'vector	
		double precision, INTENT(IN)  :: b(n)
		double precision, INTENT(out) :: fvec(m)
		integer, intent (out) :: info
	end subroutine
	
	subroutine pc_birch_murn(m,n,b,fvec,info)
		use all
		implicit none
		INTEGER, INTENT(IN)    :: m !no of functions
		INTEGER, INTENT(IN)    :: n !size o'vector	
		double precision, INTENT(IN)  :: b(n)
		double precision, INTENT(out) :: fvec(m)
		integer, intent (out) :: info
	end subroutine

	subroutine fix_pc_birch_murn(m,n,b,fvec,info)
		use all
		implicit none
		INTEGER, INTENT(IN)    :: m !no of functions
		INTEGER, INTENT(IN)    :: n !size o'vector	
		double precision, INTENT(IN)  :: b(n)
		double precision, INTENT(out) :: fvec(m)
		integer, intent (out) :: info
	end subroutine
	
	subroutine bm_jac (m,n,y,fvec,fjac,ldfjac,iflag)
		use all
		implicit none
		INTEGER, INTENT(IN)    :: m !no of functions n_all
		INTEGER, INTENT(IN)    :: n !no of parameters 3
		INTEGER, INTENT(IN)    :: ldfjac !equivalent to m	
		INTEGER, INTENT(INOUT) :: iflag !equivalent to m		
		double precision, INTENT(INOUT)  :: y(n) !parameters
		double precision, INTENT(INOUT)  :: fjac(ldfjac,n) 
		double precision, INTENT(out) :: fvec(m) !residuals
		integer x
	end subroutine

	subroutine bm_2_jac (m,n,y,fvec,fjac,ldfjac,iflag)
		use all
		implicit none
		INTEGER, INTENT(IN)    :: m !no of functions n_all
		INTEGER, INTENT(IN)    :: n !no of parameters 3
		INTEGER, INTENT(IN)    :: ldfjac !equivalent to m	
		INTEGER, INTENT(INOUT) :: iflag !equivalent to m		
		double precision, INTENT(INOUT)  :: y(n) !parameters
		double precision, INTENT(INOUT)  :: fjac(ldfjac,n) 
		double precision, INTENT(out) :: fvec(m) !residuals
		integer x
	end subroutine	

	subroutine bm_3_jac (m,n,y,fvec,fjac,ldfjac,iflag)
		use all
		implicit none
		INTEGER, INTENT(IN)    :: m !no of functions n_all
		INTEGER, INTENT(IN)    :: n !no of parameters 3
		INTEGER, INTENT(IN)    :: ldfjac !equivalent to m	
		INTEGER, INTENT(INOUT) :: iflag !equivalent to m		
		double precision, INTENT(INOUT)  :: y(n) !parameters
		double precision, INTENT(INOUT)  :: fjac(ldfjac,n) 
		double precision, INTENT(out) :: fvec(m) !residuals
		integer x
	end subroutine

	subroutine bm_3_jac_pc (m,n,y,fvec,fjac,ldfjac,iflag)
		use all
		implicit none
		INTEGER, INTENT(IN)    :: m !no of functions n_all
		INTEGER, INTENT(IN)    :: n !no of parameters 3
		INTEGER, INTENT(IN)    :: ldfjac !equivalent to m	
		INTEGER, INTENT(INOUT) :: iflag !equivalent to m		
		double precision, INTENT(INOUT)  :: y(n) !parameters
		double precision, INTENT(INOUT)  :: fjac(ldfjac,n) 
		double precision, INTENT(out) :: fvec(m) !residuals
		integer x
	end subroutine
	
	subroutine bm_3_jac_pc_fix (m,n,y,fvec,fjac,ldfjac,iflag)
		use all
		implicit none
		INTEGER, INTENT(IN)    :: m !no of functions n_all
		INTEGER, INTENT(IN)    :: n !no of parameters 3
		INTEGER, INTENT(IN)    :: ldfjac !equivalent to m	
		INTEGER, INTENT(INOUT) :: iflag !equivalent to m		
		double precision, INTENT(INOUT)  :: y(n) !parameters
		double precision, INTENT(INOUT)  :: fjac(ldfjac,n) 
		double precision, INTENT(out) :: fvec(m) !residuals
		integer x
	end subroutine


	
	END INTERFACE
	
	id=0
	id(1,1)=1
	id(2,2)=1
	id(3,3)=1	

	pi=dacos(0d0)*2
	pconv=1e3

	call get_command_argument(1,input)
	write (*,*) input
	open(file=input, unit=8)
	
	read(8,*) n_all,TorP,pt,ref_pt,Pc,finite,euler
	
	write(*,*) 'N'
	write(*,*) N_all
	write(*,*) 'TorP'
	write(*,*) TorP
	allocate(a(6,n_all))
	allocate(TP(n_all))
	allocate(TP_err(n_all))
	allocate(TP_err_spare(n_all))
	allocate(Vol(n_all))
	allocate(prax(3,3,n_all))
	allocate(K(3,n_all))
	allocate(strain(3,n_all))
	allocate(Jac(4,n_all))	
	allocate(err_K(3,n_all))	
	allocate(covar(4,4))
	
	write (*,*) 'INPUTS',n_all,TorP,pt,ref_pt,Pc,Finite,euler
	
	do i=1,n_all
		read(8,*) TP(i),TP_err(i),a(:,i)
		a(4,i)=a(4,i)/180d0*pi
		a(5,i)=a(5,i)/180d0*pi
		a(6,i)=a(6,i)/180d0*pi

		if(TP(i).eq.0) then
			TP(i)=1e-32
		endif

		if((TP_err(i)/TP(i)).lt.1e-10) then
			call exit(2)
		endif
		Vol(i)=cellvol(a(:,i))
		call orthmat(e,a(:,i))

		if(i.eq.1)then
			if(euler.eq.'Y') then
				call DGETRF( 3, 3, e, 3, IPIV, info ) !LAPACK FIND INVERSE
				call DGETRI( 3, e, 3, IPIV, WORK, 192,info)
			endif
			base=e
			ep = 0
		endif

		if(euler.eq.'Y') then
			call orthmat(e,a(:,i))
			ep=id-MatMul(base,e)
		else
			call DGETRF( 3, 3, e, 3, IPIV, info ) !LAPACK FIND INVERSE
			call DGETRI( 3, e, 3, IPIV, WORK, 192,info)
			ep=MatMul(e,base)-id
		endif
		
		if(i.eq.1)then
			ep=0
		endif		

		
		
		prax(:,:,i)=(ep+transpose(ep))/2
		if(finite.eq.'Y') then
			prax(:,:,i)=(ep+transpose(ep)+MatMul(ep,transpose(ep)))/2			
		endif
		ep = prax(:,:,i)

		call orthmat(e,a(:,i))
		call DSYEV( 'V', 'U', 3, prax(:,:,i), 3, strain(:,i), WORK, 102, info )
	enddo
	write (*,*) n_all,TorP,pt,ref_pt,Pc	
	if (Pc .eq. -1) then !If the pressure was left unspecified then set it to the lowest pressure.
		Pc = minval(TP)
	endif
	
	write (*,*) n_all,TorP,pt,ref_pt,Pc
!PRINT TP
	write (*,*) 'TP'
	write(*,*) TP
!PRINT STRAIN
	write (*,*) 'STRAIN'
	do i=1,n_all
		write(*,*) TP(i),strain(:,i)*1e2
	enddo
!DONE PRINT STRAIN
!PRINT STRAIN
	write (*,*) 'VOL'
	do i=1,n_all
		write(*,*) TP(i),Vol(i)
	enddo
!DONE PRINT STRAIN


!CALCULATE PRINCIPAL AXES
	write (*,*) 'PRAX'
	u=n_all/2+1
	call orthmat(e,a(:,u))
	

	ep = MatMul(e,prax(:,:,u))
	write(*,*) Transpose(ep)
	
	call DGETRF( 3, 3, ep, 3, IPIV, info ) !LAPACK FIND INVERSE
	call DGETRI( 3, ep, 3, IPIV, WORK, 192,info)
	write (*,*) 'CRAX'
	write(*,*) ep
	
	write (*,*) 'OrthoPRAX'
	write(*,*) prax(:,:,u)
	
!LINEAR STRAIN FITTING
	write(*,*) 'LINLINFIT'
err_alp=0
err_dep=0
	do i=1,3	
		call linreg(TP,strain(i,:),n_all,alpha(i),dep(i),err_alp(i),err_dep(i)) !Carry out linear regression -> although incorrect for P expansions, important as starting point for least squares minimising
		
		call errlinreg(n_all,TP,strain(i,:),TP_err,dep(i),alpha(i),err_dep(i),err_alp(i))
		!errlinreg(n,x,y,y_err,a,b,sig_a,sig_b)
!alpha is gradient, dep offset	
		write(*,*)	i,alpha(i),dep(i),err_alp(i),err_dep(i)
	enddo
	if((TorP.eq.'P').and.(n_all .ge. 4) )then	
	!! LEAST SQUARES FITTING TO pressure equation l=dl+lam*(p-pc)^nu w/pc -ve 
	!! b are the coefficients, K is the residual
	!! GRID SEARCH
	best = 1e6
	do it=1,grid
	write (*,*) it
	do jt=1,grid	
	do kt=1,grid
	do lt=1,grid
		
		it_fac(1)=50d0**(it/ dble(grid))
		it_fac(2)=50d0**(jt/ dble(grid))
		it_fac(3)=MINVAL(TP)*(kt/ dble(grid))
		it_fac(4)=2d0*(lt/ dble(grid))
		
		do i=1,3 !initialise Linear guess
			b(i,1)=dep(i)*it_fac(1)/10d0!Setting dl
			b(i,2)=alpha(i)*it_fac(2)/10d0!Setting lambda
			b(i,3)=0+it_fac(3)!Setting Pc
			b(i,4)=0+it_fac(4)!Setting nu			
		enddo
		K=0 
		do i_all=1,3
	
			call lmdif1 ( press_fit, n_all, 4, b(i_all,:), K(i_all,:), 1d-7, info )
			if(SUM( K(i_all,:)).lt.best(i_all)) then
				tp_err_spare =  tp_err
				tp_err=1
				call press_fit(n_all,4,b(i_all,:), K(i_all,:),info)
				tp_err=tp_err_spare	
				best(i_all)=SUM( K(i_all,:))	
				f0=0
				do i=1,n_all
					Jac(1,i)=1
					Jac(2,i)=(TP(i)-b(i_all,3))**b(i_all,4)
					Jac(3,i)=-b(i_all,2)*b(i_all,4)*(TP(i)-b(i_all,3))**(b(i_all,4)-1)
					Jac(4,i)=(dlog(TP(i)-b(i_all,3)))*((TP(i)-b(i_all,3))**b(i_all,4))*b(i_all,2)	
					f0 = f0+K(i_all,i)
				enddo	
				
				covar=Matmul(Jac,Transpose(Jac))
				call DGETRF( 4, 4, covar, 4, i4piv, info ) !LAPACK FIND INVERSE
				call DGETRI( 4, covar, 4, i4piv, WORK, 192,info)
				covar=covar*f0/(n_all-4)!/n_all
	
				do i=1,n_all
					Jac(1,i)=0
					Jac(2,i)=(TP(i)-b(i_all,3))**(b(i_all,4)-1)*b(i_all,4)
					Jac(3,i)=-b(i_all,2)*b(i_all,4)*(b(i_all,4)-1)*(TP(i)-b(i_all,3))**(b(i_all,4)-2)
					Jac(4,i)=(b(i_all,4)*dlog(TP(i)-b(i_all,3))+1)*((TP(i)-b(i_all,3))**(b(i_all,4)-1))*b(i_all,2)	
				enddo	
				err_K(i_all,:)=0
				do u=1,n_all
					do i=1,4
						do j=1,4
							err_K(i_all,u)=err_K(i_all,u)+Jac(i,u)*Jac(j,u)*covar(i,j)
						enddo
					enddo
					err_K(i_all,u)=err_K(i_all,u)**0.5d0
				enddo
				best_b(i_all,:)=b(i_all,:)
			endif
		enddo
		
		enddo
		enddo
		enddo
		enddo	

		write(*,*) 'LINPRESSFIT'		
		do i = 1,3
			write(*,*) i,best_b(i,:)		
		enddo
		
		write(*,*) 'K'			
		do  u=1,n_all      
			do i=1,3
				K(i,u)=-1*best_b(i,4)*best_b(i,2)*( (TP(u)-best_b(i,3))** (best_b(i,4)-1) )
			enddo		
			write(*,*) TP(u),PConv*K(:,u),PConv*err_k(:,u)			
		enddo
		do i=1,3
			temp(i)=-best_b(i,2)*((TP(n_all)-best_b(i,3))**best_b(i,4)-(TP(1)-best_b(i,3))**best_b(i,4))/(TP(n_all)-TP(1))
		enddo		
		write(*,*) 'MEANK'	
		write(*,*) temp*pconv

		write(*,*) 'FIT',best(3),'param',b(3,:)

		
		

	endif
	write(*,*) 'just before bulk fit'
!BULK STRAIN FITTING
	write(*,*) 'BULKLINFIT'
!	call linreg(TP,Vol,n_all,alpha(1),dep(1),err_alp(1),err_dep(1)) !Carry out linear regression -> although incorrect for P expansions, important as starting point for least squares minimising
!alpha is gradient, dep offset	
TP_err = 1
	call errlinreg(n_all,Vol,TP,TP_err,dep(1),alpha(1),err_dep(1),err_alp(1))
		!errlinreg(n,x,y,y_err,a,b,sig_a,sig_b)
!write(*,*)	alpha(1),dep(1),err_alp(1),err_dep(1) !
	write(*,*)	1/alpha(1),-dep(1)/alpha(1),err_alp(1)/(alpha(1)**2),err_dep(1) !

	
	if(TorP.eq.'P') then
		deallocate(Jac)	
		deallocate(covar)
		allocate(Jac(n_all,2))	
		allocate(covar(2,2))		
!SECOND ORDER BM FITTING
		bm(1)=100
		bm(2)=Vol(1)
		bm(3)=4
		bm(4)=pc
		
		call lmder1 (bm_2_jac, n_all, 2,bm, K(1,:),jac,n_all, 1d-12, info,K(2,:),work,192)
		tp_err_spare =  tp_err
		tp_err=1
		info = 2
		call bm_2_jac (n_all,2,bm,K(1,:),jac,n_all,info)
		info = 1
		call bm_2_jac (n_all,2,bm,K(1,:),jac,n_all,info)		
		K(1,:)=K(1,:)**2
		covar=Matmul(Transpose(Jac),Jac)
		call DGETRF( 2, 2, covar, 2, i2piv, info ) !LAPACK FIND INVERSE
		call DGETRI( 2, covar, 2, i2piv, WORK, 192,info)
		covar=covar*sum(K(1,:))/(n_all-2)
		tp_err=tp_err_spare
		write(*,*) 'BM2'
		write(*,*) BM(1),BM(2)
		write(*,*) covar


!BIRCH-MURNAGHAN FITTING
	if (n_all .ge. 3) then
		K=0
		deallocate(Jac)	
		deallocate(covar)
		allocate(Jac(n_all,3))	
		allocate(covar(3,3))	
		call lmder1 (bm_3_jac, n_all, 3,bm, K(1,:),jac,n_all, 1d-12, info,K(2,:),work,192 )
		tp_err_spare =  tp_err
		tp_err=1
		info = 2
		call bm_3_jac (n_all,3,bm,K(1,:),jac,n_all,info)
		info = 1
		call bm_3_jac (n_all,3,bm,K(1,:),jac,n_all,info)		
		K(1,:)=K(1,:)**2
		covar=Matmul(Transpose(Jac),Jac)
		call DGETRF( 3, 3, covar, 3, ipiv, info ) !LAPACK FIND INVERSE
		call DGETRI( 3, covar, 3, ipiv, WORK, 192,info)
		covar=covar*sum(K(1,:))/(n_all-3)
		tp_err=tp_err_spare		
		write(*,*) 'BM'
		write(*,*) BM(1),BM(2),BM(3)
		write(*,*) covar
	endif
!BM w/ critical Pressure
	if ((pt.eq.'Y').and.(n_all .ge. 4)) then
			K=0
			write(*,*) 'BMPC'
			call lmder1 (bm_3_jac_pc_fix, n_all, 3,bm, K(1,:),jac,n_all, 1d-12, info,K(2,:),work,192 )
			tp_err_spare =  tp_err
			tp_err=1
			write(*,*) bm
			info = 2
			call bm_3_jac_pc_fix (n_all,3,bm,K(1,:),jac,n_all,info)
			info = 1
			call bm_3_jac_pc_fix (n_all,3,bm,K(1,:),jac,n_all,info)		
			K(1,:)=K(1,:)**2
			covar=Matmul(Transpose(Jac),Jac)
			call DGETRF( 3, 3, covar, 3, ipiv, info ) !LAPACK FIND INVERSE
			call DGETRI( 3, covar, 3, ipiv, WORK, 192,info)
			covar=covar*sum(K(1,:))/(n_all-4)
			tp_err=tp_err_spare		
			write(*,*) covar
		endif
		deallocate(Jac)	
		deallocate(covar)
	
!This is the Birch-Murnaghan fitting for strain. The code simply recycles the volume code with volume redefined
!Commented out as not needed
!		do i=1,3 !each direction
!			Vol=(strain(i,:)+1)**3d0
!			allocate(Jac(n_all,2))	
!			allocate(covar(2,2))		
!	!SECOND ORDER BM FITTING
!			bm(1)=0
!			bm(2)=Vol(1)
!			bm(3)=4
!			bm(4)=pc
!			!write(*,*) i,Vol
!			call lmder1 (bm_2_jac, n_all, 2,bm, K(1,:),jac,n_all, 1d-12, info,K(2,:),work,192)
!			tp_err_spare =  tp_err
!			tp_err=1
!			info = 2
!			call bm_2_jac (n_all,2,bm,K(1,:),jac,n_all,info)
!			info = 1
!			call bm_2_jac (n_all,2,bm,K(1,:),jac,n_all,info)		
!			K(1,:)=K(1,:)**2
!			covar=Matmul(Transpose(Jac),Jac)
!			call DGETRF( 2, 2, covar, 2, i2piv, info ) !LAPACK FIND INVERSE
!			call DGETRI( 2, covar, 2, i2piv, WORK, 192,info)
!			covar=covar*sum(K(1,:))/(n_all-2)
!			tp_err=tp_err_spare
!			write(*,'(A9,I1)') 'BM2STRAIN',i
!			write(*,*) BM(1),BM(2)
!			write(*,*) covar
!	
!	
!	!BIRCH-MURNAGHAN FITTING
!		if (n_all .ge. 3) then
!			K=0
!			deallocate(Jac)	
!			deallocate(covar)
!			allocate(Jac(n_all,3))	
!			allocate(covar(3,3))	
!			call lmder1 (bm_3_jac, n_all, 3,bm, K(1,:),jac,n_all, 1d-12, info,K(2,:),work,192 )
!			tp_err_spare =  tp_err
!			tp_err=1
!			info = 2
!			call bm_3_jac (n_all,3,bm,K(1,:),jac,n_all,info)
!			info = 1
!			call bm_3_jac (n_all,3,bm,K(1,:),jac,n_all,info)		
!			K(1,:)=K(1,:)**2
!			covar=Matmul(Transpose(Jac),Jac)
!			call DGETRF( 3, 3, covar, 3, ipiv, info ) !LAPACK FIND INVERSE
!			call DGETRI( 3, covar, 3, ipiv, WORK, 192,info)
!			covar=covar*sum(K(1,:))/(n_all-3)
!			tp_err=tp_err_spare		
!			write(*,'(A8,I1)') 'BMSTRAIN',i
!			write(*,*) BM(1),BM(2),BM(3)
!			write(*,*) covar
!		endif
!	!BM w/ critical Pressure
!		if ((pt.eq.'Y').and.(n_all .ge. 4)) then
!				K=0
!				write(*,'(A10,I1)') 'BMPCSTRAIN',i
!				call lmder1 (bm_3_jac_pc_fix, n_all, 3,bm, K(1,:),jac,n_all, 1d-12, info,K(2,:),work,192 )
!				tp_err_spare =  tp_err
!				tp_err=1
!				write(*,*) bm
!				info = 2
!				call bm_3_jac_pc_fix (n_all,3,bm,K(1,:),jac,n_all,info)
!				info = 1
!				call bm_3_jac_pc_fix (n_all,3,bm,K(1,:),jac,n_all,info)		
!				K(1,:)=K(1,:)**2
!				covar=Matmul(Transpose(Jac),Jac)
!				call DGETRF( 3, 3, covar, 3, ipiv, info ) !LAPACK FIND INVERSE
!				call DGETRI( 3, covar, 3, ipiv, WORK, 192,info)
!				covar=covar*sum(K(1,:))/(n_all-4)
!				tp_err=tp_err_spare		
!				write(*,*) covar
!		endif
!			deallocate(Jac)	
!			deallocate(covar)
!		enddo	
	endif

	write(*,*) 'gas',acos((cos(a(4,2))*cos(a(5,2))-cos(a(6,2)))/(sin(a(4,2))*sin(a(5,2))))
	deallocate (K)
	deallocate (strain)
	deallocate (prax)	
	deallocate (a)	
	deallocate(TP)
	deallocate(TP_err)
	deallocate(TP_err_spare)	
	deallocate(Vol)	
	deallocate(Jac)	
	deallocate(err_K)		
	close(11)	
	close(10)	
	close (9)
	close(8)
	write(*,*) 'END OF FORTRAN'
end

subroutine orthmat(mat,latt)
	double precision mat(3,3)
	double precision latt(6),ga_s
	intent(in) latt
	intent(out) mat
	ga_s = acos((cos(latt(4))*cos(latt(5))-cos(latt(6)))/(sin(latt(4))*sin(latt(5))))
	!write(*,*) gas,ga_s/acos(-1d0),acos(-1d0)
	mat(1,1)=1/(latt(1)*sin(latt(5))*sin(ga_s))
	mat(1,2)=0
	mat(1,3)=0
	mat(2,1)=cos(ga_s)/(latt(2)*sin(latt(4))*sin(ga_s))
	mat(2,2)=1/(latt(2)*sin(latt(4)))
	mat(2,3)=0
	mat(3,1)=(cos(latt(4))*cos(ga_s)/sin(latt(4))+cos(latt(5))/sin(latt(5)))/(-1*latt(3)*sin(ga_s))
	mat(3,2)=-1*cos(latt(4))/(latt(3)*sin(latt(4)))
	mat(3,3)=1/latt(3)	

!	ga_s = acos((cos(latt(4))*cos(latt(6))-cos(latt(5)))/((sin(latt(4))*sin(latt(6)))))
!	
!	mat(1,1)=1/(latt(1)*sin(latt(6))*sin(ga_s))
!	mat(2,1)=1/(latt(2)*tan(latt(4))*tan(ga_s))-1/(latt(2)*tan(latt(6))*sin(ga_s))
!	mat(3,1)=1/(latt(3)*sin(latt(4))*tan(ga_s))!*(-1)
!	mat(1,2)=0
!	mat(2,2)=1/latt(2)
!	mat(3,2)=0
!	mat(1,3)=0
!	mat(2,3)=-1/(latt(2)*tan(latt(4)))
!	mat(3,3)=1/(latt(3)*sin(latt(4)))

!	ga_s = acos(cos(latt(4))*cos(latt(5))-cos(latt(6)))/(sin(latt(4))*sin(latt(5)))
!	mat(1,1) = latt(1)*sin(latt(5))*sin(ga_s)
!	mat(1,2) = 0
!	mat(1,3) = 3
!	mat(2,1) = -latt(1)*sin(latt(5))*cos(ga_s)
!	mat(2,2) = latt(2)*sin(latt(4))
!	mat(2,3) = 0
!	mat(3,1) = latt(1)*cos(latt(5))
!	mat(3,1) = latt(2)*cos(latt(4))
!	mat(3,1) = latt(3)	
	
!	ga_s = acos((cos(latt(5))*cos(latt(6))-cos(latt(4)))/((sin(latt(5))*sin(latt(6)))))
!	
!	mat(1,1)=1/(latt(1)*sin(latt(6))*sin(ga_s))
!	mat(2,1)=1/(latt(2)*tan(latt(4))*tan(ga_s))-1/(latt(2)*tan(latt(6))*sin(ga_s))
!	mat(3,1)=1/(latt(3)*sin(latt(4))*tan(ga_s))!*(-1)
!	mat(1,2)=0
!	mat(2,2)=1/latt(2)
!	mat(3,2)=0
!	mat(1,3)=0
!	mat(2,3)=-1/(latt(2)*tan(latt(4)))
!	mat(3,3)=1/(latt(3)*sin(latt(4)))	
!	
	
end subroutine

double precision function cellvol(latt)
	double precision latt(6)
	intent (in) latt
	cellvol =  latt(1)*latt(2)*latt(3)*(1-cos(latt(4))**2-cos(latt(5))**2&
	-cos(latt(6))**2+2*cos(latt(4))*cos(latt(5))*cos(latt(6)))**(0.5d0)
end function

subroutine linreg(TP,strain,n,alpha,dep,err_alp,err_dep)
	implicit none
	integer i,j,k,n
	double precision TP(n)
	double precision strain(n)
	double precision alpha
	double precision dep,err_alp,err_dep
	
	double precision Sxx,Sx,Sy,Syy,Sxy,err_S
	
	intent(in) TP,strain,n
	intent(out) alpha,dep,err_alp,err_dep
	Sx=0
	Sxx=0
	Sxy=0
	Syy=0
	Sy=0
	do k=1,n		
		Sx=Sx+TP(k)
		Sxx=Sxx+TP(k)**2
		Sxy=Sxy+TP(k)*strain(k)
		Syy=Syy+strain(k)**2
		Sy=Sy+strain(k)
	enddo		
	alpha=(n*Sxy-Sx*Sy)/(n*Sxx-(Sx)**2)
	dep=Sy/n-alpha*Sx/n
	
	err_S = (n*Syy-Sy**2-(alpha**2)*(n*Sxx-Sx**2))/dble((n-2)*n)
	err_alp = (n*err_S)/(n*Sxx-Sx**2)
	err_dep = err_alp*Sxx/(dble(n))

	err_alp = err_alp**0.5d0
	err_dep = err_dep**0.5d0
end subroutine	

subroutine errlinreg(n,x,y,y_err,a,b,sig_a,sig_b)
	implicit none
	integer n
	double precision a,b,sig_a,sig_b,x(n),y(n),y_err(n),del,sig
	double precision D(2,n),U(n),Var(2,2)

	intent(in) n,x,y,y_err
	intent(out) a,b,sig_a,sig_b

	Del = sum(y_err**(-2d0))*sum((x/y_err)**2d0)-(sum(x/(y_err**2d0))**2d0)
	
	a = (sum((x/y_err)**2d0)*sum(y/(y_err**2d0))-sum(x/(y_err**2d0))*sum(x*y/(y_err**2d0)))/Del
	b = (sum(x*y/(y_err**2d0))*sum(1/(y_err**2d0))-sum(x/(y_err**2d0))*sum(y/(y_err**2d0)))/Del
	
	U = y-(a+b*x)

	sig_a = (sum(x**2)**2*sum(U**2) - 2*sum(x)*sum(x**2)*sum(U**2*x) + sum(x)**2*sum(U**2*x**2))/(n*Sum(x**2)-sum(x)**2)**2
	sig_b =(sum(X)**2*sum(U**2) - 2*n*sum(X)*sum(U**2*X) + n**2*sum(U**2*X**2))/(n*Sum(x**2)-sum(x)**2)**2
	sig_a = sqrt(sig_a)
	sig_b = sqrt(sig_b)
end subroutine		



subroutine press_fit(m,n,y,fvec,info)
	use all
	implicit none
	INTEGER, INTENT(IN)    :: m !no of functions n_all
	INTEGER, INTENT(IN)    :: n !size o'vector	i.e. no of parameters 4
	double precision, INTENT(IN)  :: y(n)
	double precision, INTENT(out) :: fvec(m)
	integer, intent (out) :: info
	integer x
	do x=1,m
		fvec(x)=(strain(i_all,x)-y(1)-y(2)*((TP(x)-y(3))**2)**(y(4)/2d0))**2/(TP_err(x)**2) !ERROR WEIGHTING IS DIVISION BY
		if(y(3).gt. MINVAL(TP) )fvec(x)=1e9
	enddo	
end subroutine

subroutine birch_murn(m,n,y,fvec,info)
	use all
	implicit none
	INTEGER, INTENT(IN)    :: m !no of functions n_all
	INTEGER, INTENT(IN)    :: n !no of parameters 3
	
	double precision, INTENT(IN)  :: y(n)
	double precision, INTENT(out) :: fvec(m)
	integer, intent (out) :: info
	double precision temp,temp1,temp2,temp3,temp4
	integer x
	do x=1,m
		temp = (y(2)/Vol(x))**(1d0/3d0)
		temp1 = (3d0*y(1)/2d0)
		temp2 = (temp**7d0-temp**5d0)
		temp3 = (3d0/4d0)*(y(3)-4)
		temp4 = (temp**2d0-1)
		fvec(x) = (TP(x) - temp1*temp2*(1+temp3*temp4))**2
	enddo
end subroutine

subroutine bm_jac (m,n,y,fvec,fjac,ldfjac,iflag)
	use all
	implicit none
	INTEGER, INTENT(IN)    :: m !no of functions n_all
	INTEGER, INTENT(IN)    :: n !no of parameters 3
	INTEGER, INTENT(IN)    :: ldfjac !equivalent to m	
	INTEGER, INTENT(INOUT) :: iflag !equivalent to m		
	double precision, INTENT(INOUT)  :: y(n) !parameters
	double precision, INTENT(INOUT)  :: fjac(ldfjac,n) 
	double precision, INTENT(out) :: fvec(m) !residuals
	integer x
	if(iflag.eq.1) then
		do x=1,m
			fvec(x) = (TP(x)-(3*y(1)*(-(y(2)/Vol(x))**1.6666666666666667+(y(2)/Vol(x))**2.3333333333333335)&
			*(1+(3*(-4+y(3))*(-1+(y(2)/Vol(x))**0.6666666666666666))/4.))/2.)**2
		enddo
	endif
	if(iflag.eq.2) then
		do x=1,m
			fjac(x,1)= (3*(y(2)/Vol(x))**1.6666666666666667*(-1 + (y(2)/Vol(x))**0.6666666666666666)*&
			(1 + (3*(-4 + y(3))*(-1 + (y(2)/Vol(x))**0.6666666666666666))/4.))/2.
			fjac(x,2)=(y(1)*y(2)*(27*(-4 + y(3))*y(2)**2 + 14*(14 - 3*y(3))*Vol(x)*y(2)*(y(2)/Vol(x))**0.3333333333333333&
			+ 5*(-16 + 3*y(3))*Vol(x)**2*(y(2)/Vol(x))**0.6666666666666666))/(8.*Vol(x)**4)
			fjac(x,3)=(9*y(1)*y(2)*(y(2) - Vol(x)*(y(2)/Vol(x))**0.3333333333333333)**2)/(8.*Vol(x)**3)
		enddo
	endif	
end subroutine

subroutine pc_birch_murn(m,n,y,fvec,info)
	use all
	implicit none
	INTEGER, INTENT(IN)    :: m !no of functions n_all
	INTEGER, INTENT(IN)    :: n !no of parameters 4 b0,v0,b'0,pc
	
	double precision, INTENT(IN)  :: y(n)
	double precision, INTENT(out) :: fvec(m)
	integer, intent (out) :: info
	double precision temp,temp1,temp2,temp3,maxP
	integer x
	maxP = minval(TP)*(1.000001d0)
	
	do x=1,m
		temp = (y(2)/Vol(x))**(1d0/3d0)		
		temp1 = (3d0*y(1)-5d0*y(4))/2d0
		temp2 = (9d0*y(1)/8d0)*(y(3)-4+(35d0*y(4))/(9d0*y(1)))
		temp3 = (1-temp**2d0)
		fvec(x)=( TP(x)-(temp**5d0)*(y(4)-temp1*temp3+temp2*(temp3**2d0)) )**2
		if (y(4).gt.maxP) then 
			fvec(x) = 1e9
		endif
		if (y(4).lt.0) then 
			fvec(x) = 1e9
		endif

	enddo
end subroutine

subroutine fix_pc_birch_murn(m,n,y,fvec,info)
	use all
	implicit none
	INTEGER, INTENT(IN)    :: m !no of functions n_all
	INTEGER, INTENT(IN)    :: n !no of parameters 4 b0,v0,b'0,pc
	
	double precision, INTENT(IN)  :: y(n)
	double precision, INTENT(out) :: fvec(m)
	integer, intent (out) :: info
	double precision temp,temp1,temp2,temp3,maxP
	integer x
	do x=1,m
		temp = (y(2)/Vol(x))**(1d0/3d0)		
		temp1 = (3d0*y(1)-5d0*Pc)/2d0
		temp2 = (9d0*y(1)/8d0)*(y(3)-4+(35d0*Pc)/(9d0*y(1)))
		temp3 = (1-temp**2d0)
		fvec(x)=( TP(x)-(temp**5d0)*(Pc-temp1*temp3+temp2*(temp3**2d0)) )**2
	enddo
end subroutine

subroutine sec_birch_murn(m,n,y,fvec,info)
	use all
	implicit none
	INTEGER, INTENT(IN)    :: m !no of functions n_all
	INTEGER, INTENT(IN)    :: n !no of parameters 2
	
	double precision, INTENT(IN)  :: y(n)
	double precision, INTENT(out) :: fvec(m)
	integer, intent (out) :: info
	double precision temp
	integer x
	do x=1,m
		temp = (y(2)/Vol(x))**(1d0/3d0)
		fvec(x)=(TP(x)-(3d0*y(1)/2d0)*(temp**7d0-temp**5d0))**2
	enddo
end subroutine

subroutine debye (m,n,y,fvec,info)
	use all
	implicit none
	INTEGER, INTENT(IN)    :: m !no of functions n_all
	INTEGER, INTENT(IN)    :: n !no of parameters 3
	
	double precision, INTENT(IN)  :: y(n)
	double precision, INTENT(out) :: fvec(m)
	integer, intent (out) :: info
	integer x
	
end subroutine

subroutine bm_3_jac (m,n,y,fvec,fjac,ldfjac,iflag)
	use all
	implicit none
	INTEGER, INTENT(IN)    :: m !no of functions n_all
	INTEGER, INTENT(IN)    :: n !no of parameters 3
	INTEGER, INTENT(IN)    :: ldfjac !equivalent to m	
	INTEGER, INTENT(INOUT) :: iflag !equivalent to m		
	double precision, INTENT(INOUT)  :: y(n) !parameters
	double precision, INTENT(inOUT)  :: fjac(ldfjac,n) 
	double precision, INTENT(inout) :: fvec(m) !residuals
	integer x
	if(iflag.eq.1) then
		do x=1,m
			fvec(x) = (TP(x)-(3*y(1)*(-(y(2)/Vol(x))**1.6666666666666667+(y(2)/Vol(x))**2.3333333333333335)&
			*(1+(3*(-4+y(3))*(-1+(y(2)/Vol(x))**0.6666666666666666))/4.))/2.)
		enddo
	endif
	if(iflag.eq.2) then
		do x=1,m
			fjac(x,1)= -(3*(y(2)/Vol(x))**1.6666666666666667*(-1 + (y(2)/Vol(x))**0.6666666666666666)*&
			(1 + (3*(-4 + y(3))*(-1 +(y(2)/Vol(x))**0.6666666666666666))/4.))/2.
			fjac(x,2)=-(y(1)*(27*(-4 + y(3))*y(2)**2 + 14*(14 - 3*y(3))*Vol(x)*y(2)*(y(2)/Vol(x))**0.3333333333333333 + &
			5*(-16 + 3*y(3))*Vol(x)**2*(y(2)/Vol(x))**0.6666666666666666))/(8.*Vol(x)**3)
			fjac(x,3)=-(9*y(1)*y(2)*(y(2) - Vol(x)*(y(2)/Vol(x))**0.3333333333333333)**2)/(8.*Vol(x)**3)
		enddo
	endif	
end subroutine

subroutine bm_2_jac (m,n,y,fvec,fjac,ldfjac,iflag)
	use all
	implicit none
	INTEGER, INTENT(IN)    :: m !no of functions n_all
	INTEGER, INTENT(IN)    :: n !no of parameters 3
	INTEGER, INTENT(IN)    :: ldfjac !equivalent to m	
	INTEGER, INTENT(INOUT) :: iflag !equivalent to m		
	double precision, INTENT(INOUT)  :: y(n) !parameters
	double precision, INTENT(inOUT)  :: fjac(ldfjac,n) 
	double precision, INTENT(inout) :: fvec(m) !residuals
	integer x
	if(iflag.eq.1) then
		do x=1,m
			fvec(x)=TP(x)-3*y(1)*(y(2)/Vol(x))**1.6666666666666667*(-1+(y(2)/Vol(x))**0.6666666666666666)/2d0
		enddo
	endif
	if(iflag.eq.1) then
		do x=1,m
			fjac(x,1)=-3*(y(2)/Vol(x))**1.6666666666666667*(-1 + (y(2)/Vol(x))**0.6666666666666666)/2d0
			fjac(x,2)=-(y(1)*(7*y(2)*(y(2)/Vol(x))**0.3333333333333333 - 5*Vol(x)*(y(2)/Vol(x))**0.6666666666666666))/Vol(x)**2/2d0
		enddo
	endif	
end subroutine

subroutine bm_3_jac_pc (m,n,y,fvec,fjac,ldfjac,iflag)
	use all
	implicit none
	INTEGER, INTENT(IN)    :: m !no of functions n_all
	INTEGER, INTENT(IN)    :: n !no of parameters 4
	INTEGER, INTENT(IN)    :: ldfjac !equivalent to m	
	INTEGER, INTENT(INOUT) :: iflag !equivalent to m		
	double precision, INTENT(INOUT)  :: y(n) !parameters
	double precision, INTENT(inOUT)  :: fjac(ldfjac,n) 
	double precision, INTENT(inout) :: fvec(m) !residuals
	integer x
	double precision maxP
	maxP = minval(TP)*(1.000001d0)
	if(iflag.eq.1) then
		do x=1,m
			fvec(x)=TP(x)-       (y(2)/Vol(x))**1.6666666666666667*(y(4) +&
	((3*y(1) - 5*y(4))*(-1 + (y(2)/Vol(x))**0.6666666666666666))/2. +&
	((9*y(1)*(-4 + y(3)) + 35*y(4))*(-1 + (y(2)/Vol(x))**0.6666666666666666)**2)/8.)
		enddo
		if (y(4).gt.maxP) then 
			fvec = 1e9
		endif
		if (y(4).lt.0) then 
		!	fvec= 1e9
		endif
	endif
	if(iflag.eq.2) then
		do x=1,m
			fjac(x,1)=-        (3*(y(2)/Vol(x))**1.6666666666666667*(-1 + (y(2)/Vol(x))**0.6666666666666666)*&
	    (16 - 12*(y(2)/Vol(x))**0.6666666666666666 + 3*y(3)*(-1 + (y(2)/Vol(x))**0.6666666666666666)))/8.
			fjac(x,2)=-        (105*y(4)*(y(2) - Vol(x)*(y(2)/Vol(x))**0.3333333333333333)**2 +&
	y(1)*(27*(-4 + y(3))*y(2)**2 + 14*(14 - 3*y(3))*Vol(x)*y(2)*(y(2)/Vol(x))**0.3333333333333333 +&
	   5*(-16 + 3*y(3))*Vol(x)**2*(y(2)/Vol(x))**0.6666666666666666))/(8.*Vol(x)**3)
			fjac(x,3)=-(9*y(1)*(y(2)/Vol(x))**1.6666666666666667*(-1 + (y(2)/Vol(x))**0.6666666666666666)**2)/8.			
			fjac(x,4)=-(y(2)/Vol(x))**1.6666666666666667*(1 + (5*(1 - (y(2)/Vol(x))**0.6666666666666666))/2. + &
         (35*(1 - (y(2)/Vol(x))**0.6666666666666666)**2)/8.)
		enddo
		if (y(4).gt.maxP) then 
			fjac(x,4) = 1e9
		endif
		if (y(4).lt.0) then 
		!	fjac(x,4)= -1e9
		endif
	endif	
end subroutine

subroutine bm_3_jac_pc_fix (m,n,y,fvec,fjac,ldfjac,iflag)
	use all
	implicit none
	INTEGER, INTENT(IN)    :: m !no of functions n_all
	INTEGER, INTENT(IN)    :: n !no of parameters 3
	INTEGER, INTENT(IN)    :: ldfjac !equivalent to m	
	INTEGER, INTENT(INOUT) :: iflag !equivalent to m		
	double precision, INTENT(INOUT)  :: y(n) !parameters
	double precision, INTENT(inOUT)  :: fjac(ldfjac,n) 
	double precision, INTENT(inout) :: fvec(m) !residuals
	integer x
	if(iflag.eq.1) then
		do x=1,m
			fvec(x)=TP(x)-       (y(2)/Vol(x))**1.6666666666666667*(pc +&
	((3*y(1) - 5*pc)*(-1 + (y(2)/Vol(x))**0.6666666666666666))/2. +&
	((9*y(1)*(-4 + y(3)) + 35*pc)*(-1 + (y(2)/Vol(x))**0.6666666666666666)**2)/8.)
		enddo
	endif
	if(iflag.eq.2) then
		do x=1,m
			fjac(x,1)=-        (3*(y(2)/Vol(x))**1.6666666666666667*(-1 + (y(2)/Vol(x))**0.6666666666666666)*&
	    (16 - 12*(y(2)/Vol(x))**0.6666666666666666 + 3*y(3)*(-1 + (y(2)/Vol(x))**0.6666666666666666)))/8.
			fjac(x,2)=-        (105*pc*(y(2) - Vol(x)*(y(2)/Vol(x))**0.3333333333333333)**2 +&
	y(1)*(27*(-4 + y(3))*y(2)**2 + 14*(14 - 3*y(3))*Vol(x)*y(2)*(y(2)/Vol(x))**0.3333333333333333 +&
	   5*(-16 + 3*y(3))*Vol(x)**2*(y(2)/Vol(x))**0.6666666666666666))/(8.*Vol(x)**3)
			fjac(x,3)=-(9*y(1)*(y(2)/Vol(x))**1.6666666666666667*(-1 + (y(2)/Vol(x))**0.6666666666666666)**2)/8.			
		enddo
	endif	
end subroutine

