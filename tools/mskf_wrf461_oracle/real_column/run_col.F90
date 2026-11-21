! One-column driver for WRF v4.7.1 MSKF_CPS (byte-unmodified module_cu_mskf.F),
! called as module_cumulus_driver.F calls it, on real forecast columns written
! by col_case.py. Same call shape as tools/mskf_wrf461_oracle/run_mskf.F90,
! with nz read at run time. Settings: dt 60 s, dx 12 km, cudt 5 min fixed step
! (stepcu 5), Morrison phases (warm_rain F, F_QR/F_QI/F_QS T), aercu_opt 0,
! kf_edrates 1, trigger 1, NCA -100 (an active column).
program run_col
  use module_cu_mskf, only: mskf_cps, mskf_init
  implicit none
  real, parameter :: g = 9.81, r_d = 287., cp = 7.*r_d/2., r_v = 461.6
  real, parameter :: ep_1 = r_v/r_d - 1., ep_2 = r_d/r_v
  real, parameter :: xlv0 = 3.15e6, xlv1 = 2370., xls0 = 2.905e6, xls1 = 259.532
  real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65, svpt0 = 273.15
  integer, parameter :: nsrc = 10
  integer :: ncol, nz, c, n, recl, uo
  real, allocatable :: rec(:,:)
  real, allocatable :: z3(:,:,:), z2(:,:)

  open(newunit=n, file='ncol.txt', status='old'); read(n, *) ncol, nz; close(n)
  recl = 10*nz + (nz+1) + 4
  allocate(rec(recl, ncol))
  open(newunit=n, file='col_f32.bin', access='stream', form='unformatted', &
       status='old', convert='little_endian')
  read(n) rec
  close(n)
  allocate(z3(1, nz+1, 1), z2(1, 1)); z3 = 0.; z2 = 0.
  call mskf_init(z3, z3, z3, z3, z3, z3, z2, z3, 2, 2, &
                 svp1, svp2, svp3, svpt0, 1, .false., .true., &
                 1, 2, 1, 2, 1, nz+1, 1, 1, 1, 1, 1, nz+1, &
                 1, 1, 1, 1, 1, nz, z3, z3)
  open(newunit=uo, file='wrf_out.bin', access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  do c = 1, ncol
    call one(rec(:, c))
  end do
  close(uo)

contains

  subroutine lv(r, f, a)
    real, intent(in) :: r(:)
    integer, intent(in) :: f
    real, intent(out) :: a(:)
    a = 0.
    a(1:nz) = r((f-1)*nz + 1 : f*nz)
  end subroutine lv

  subroutine one(r)
    real, intent(in) :: r(:)
    real, dimension(1, nz+1, 1) :: u, v, th, t, w, qv, dz8w, pcps, rho, pi3, &
         w0avg, rthcuten, rqvcuten, rqccuten, rqrcuten, rqicuten, rqscuten, &
         rqvften, rucuten, rvcuten, cldfra_dp, cldfra_sh, qc_kf, qi_kf, &
         qr_kf, qs_kf, nc_kf, ni_kf, nr_kf, ns_kf, ccn_kf, efcs, efis, efss, &
         udr_kf, ddr_kf, uer_kf, der_kf
    real, dimension(1, nz+1, 1, nsrc) :: aerocu
    real, dimension(1, 1) :: raincv, pratec, nca, cutop, cubot, ainc_frac, &
         timec_kf, zol, hfx, ust, pblh, xland
    logical, dimension(1, 1) :: cu_act_flag
    real, parameter :: s = -7777.0
    rthcuten = s; rqvcuten = s; rqccuten = s; rqrcuten = s; rqicuten = s
    rqscuten = s; rucuten = s; rvcuten = s; cldfra_dp = s; cldfra_sh = s
    qc_kf = s; qi_kf = s; qr_kf = s; qs_kf = s; nc_kf = s; ni_kf = s; nr_kf = s
    ns_kf = s; ccn_kf = s; efcs = s; efis = s; efss = s; udr_kf = s; ddr_kf = s
    uer_kf = s; der_kf = s; raincv = s; pratec = s; cutop = s; cubot = s
    ainc_frac = s; timec_kf = s; cu_act_flag = .false.
    rqvften = 0.; aerocu = 0.; xland = 1.
    call lv(r, 1, u(1,:,1)); call lv(r, 2, v(1,:,1)); call lv(r, 3, t(1,:,1))
    call lv(r, 4, th(1,:,1)); call lv(r, 5, qv(1,:,1)); call lv(r, 6, pcps(1,:,1))
    call lv(r, 7, rho(1,:,1)); call lv(r, 8, pi3(1,:,1)); call lv(r, 9, dz8w(1,:,1))
    call lv(r, 10, w0avg(1,:,1))
    w(1, 1:nz+1, 1) = r(10*nz + 1 : 10*nz + nz + 1)
    hfx(1,1) = r(11*nz+2); ust(1,1) = r(11*nz+3); pblh(1,1) = r(11*nz+4)
    zol(1,1) = r(11*nz+5)
    nca(1,1) = -100.
    call mskf_cps( &
         1, 2, 1, 2, 1, nz+1, &
         1, 1, 1, 1, 1, nz+1, &
         1, 1, 1, 1, 1, nz, &
         1, &
         60., 1, 12000., 5., .false., &
         rho, raincv, pratec, nca, &
         u, v, th, t, w, dz8w, pcps, pi3, &
         w0avg, xlv0, xlv1, xls0, xls1, cp, r_d, g, ep_1, &
         ep_2, svp1, svp2, svp3, svpt0, &
         5, cu_act_flag, .false., cutop, cubot, &
         qv, &
         .true., .true., .true., .true., .true., &
         rthcuten, rqvcuten, rqccuten, rqrcuten, &
         rqicuten, rqscuten, rqvften, &
         cldfra_dp, cldfra_sh, &
         qc_kf, qi_kf, qr_kf, qs_kf, &
         nc_kf, ni_kf, nr_kf, ns_kf, &
         ccn_kf, ainc_frac, &
         udr_kf, ddr_kf, &
         uer_kf, der_kf, &
         timec_kf, 1, &
         zol, hfx, ust, pblh, &
         aerocu, nsrc, 1.0, 0, &
         efcs, efis, efss, &
         rucuten, rvcuten, xland)
    write(uo) rthcuten(1,1:nz,1), rqvcuten(1,1:nz,1), rqccuten(1,1:nz,1), &
              rqrcuten(1,1:nz,1), rqicuten(1,1:nz,1), rqscuten(1,1:nz,1), &
              rucuten(1,1:nz,1), rvcuten(1,1:nz,1), cldfra_dp(1,1:nz,1), &
              cldfra_sh(1,1:nz,1), qc_kf(1,1:nz,1), qi_kf(1,1:nz,1), &
              udr_kf(1,1:nz,1), ddr_kf(1,1:nz,1), uer_kf(1,1:nz,1), &
              der_kf(1,1:nz,1), w0avg(1,1:nz,1), &
              raincv(1,1), pratec(1,1), nca(1,1), cutop(1,1), cubot(1,1), &
              timec_kf(1,1)
  end subroutine one
end program run_col
