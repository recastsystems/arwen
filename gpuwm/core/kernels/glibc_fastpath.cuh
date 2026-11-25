// glibc_fastpath.cuh -- float-float fast paths for the glibc binary32 libm copies.
//
// The glibc-exact logf/expf/powf copies (mynn_glibc_*, gfk_*, rlw_*, rsw_*,
// r_*) evaluate glibc's table + polynomial in binary64 and round once to
// binary32.  On GPUs with reduced-rate FP64 (GeForce: 1/64 of FP32) that
// binary64 chain dominates their cost.  These functions evaluate the SAME
// formula -- glibc's tables (sysdeps/ieee754/flt-32 e_logf_data.c,
// e_exp2f_data.c, e_powf_log2_data.c) and coefficients, split into float
// pairs -- in float-float arithmetic on the FP32 FMA, about 2^-44 relative,
// and return a result only when it is certain: the float-float value is
// farther than a declared error budget from every binary32 rounding boundary,
// so the binary64 value of the copy's formula (within the budget; this also
// covers the fma and mul+add variants among the copies, ~2^-53 apart) rounds
// to the same binary32.  Otherwise -- specials, negative/zero/subnormal
// arguments, tiny or huge results, values near a boundary -- they return
// GLIBC_FASTPATH_UNSURE and the copy runs its own binary64 code unchanged.
//
// Every float operation that must not be contracted is an explicit _rn
// intrinsic or __fmaf_rn, so --fmad=true units compute the same words.
// tests/test_glibc_fastpath_gpu.py compares each routed copy with its
// original on all 2^32 binary32 arguments (logf, expf) and on all 2^32 x for
// every constant exponent the sources pass to powf, plus random (x, y) pairs.
#ifndef GLIBC_FASTPATH_CUH
#define GLIBC_FASTPATH_CUH
// glibc's binary64 table split into float pairs: hi = float(d), lo = float(d - hi)
__device__ const float GFP_LOGF_INVC_H[16] = {0x1.661ec8p+0f, 0x1.571ed4p+0f, 0x1.4953ap+0f, 0x1.3c995cp+0f, 0x1.30d19p+0f, 0x1.25e228p+0f, 0x1.1bb4a4p+0f, 0x1.12359p+0f, 0x1.0953f4p+0f, 0x1p+0f, 0x1.e608dp-1f, 0x1.ca4b32p-1f, 0x1.b20366p-1f, 0x1.9c2d16p-1f, 0x1.886e6p-1f, 0x1.767ddp-1f};
__device__ const float GFP_LOGF_INVC_L[16] = {-0x1.81c31p-26f, 0x1.55f108p-25f, -0x1.e1fdeap-25f, -0x1.e8ff9p-25f, 0x1.910c94p-25f, -0x1.3d1c58p-26f, 0x1.434688p-25f, -0x1.eea348p-25f, 0x1.9900a8p-28f, 0.0f, -0x1.32dc2ap-28f, -0x1.fb2acp-30f, -0x1.12a064p-26f, 0x1.d0d516p-28f, 0x1.bc20f6p-28f, -0x1.5596f4p-26f};
__device__ const float GFP_LOGF_LOGC_H[16] = {-0x1.57bf78p-2f, -0x1.2bef0ap-2f, -0x1.01eae8p-2f, -0x1.b31d8ap-3f, -0x1.6574fp-3f, -0x1.1aa2bcp-3f, -0x1.a4e76cp-4f, -0x1.1973c6p-4f, -0x1.252f44p-5f, 0.0f, 0x1.aa5aa6p-5f, 0x1.c5e53ap-4f, 0x1.526e58p-3f, 0x1.bc286p-3f, 0x1.1058bcp-2f, 0x1.404306p-2f};
__device__ const float GFP_LOGF_LOGC_L[16] = {-0x1.1955bcp-31f, -0x1.f01b76p-28f, 0x1.5d8b32p-31f, -0x1.a0893ap-29f, -0x1.580eecp-28f, -0x1.e7204p-29f, -0x1.d181ccp-29f, 0x1.67b8ccp-30f, 0x1.c7bcf8p-31f, 0.0f, -0x1.06d33ep-32f, 0x1.46c5d6p-29f, -0x1.1be4ap-28f, 0x1.a448eep-28f, 0x1.140fdcp-27f, -0x1.09223ep-27f};
#define GFP_LOGF_A0_H -0x1.00ea34p-2f
#define GFP_LOGF_A0_L -0x1.171066p-27f
#define GFP_LOGF_A1_H 0x1.5575bp-2f
#define GFP_LOGF_A1_L 0x1.7c016ep-27f
#define GFP_LOGF_A2_H -0x1.fffffp-2f
#define GFP_LOGF_A2_L 0x1.beb7dcp-27f
#define GFP_LN2_1 0x1.62e4p-1f
#define GFP_LN2_2 0x1.7f7cp-20f
#define GFP_LN2_3 0x1.1cf78p-36f

// glibc's binary64 table split into float pairs: hi = float(d), lo = float(d - hi)
__device__ const float GFP_EXP2F_T_H[32] = {0x1p+0f, 0x1.059b0ep+0f, 0x1.0b5586p+0f, 0x1.11301ep+0f, 0x1.172b84p+0f, 0x1.1d4874p+0f, 0x1.2387a6p+0f, 0x1.29e9ep+0f, 0x1.306fep+0f, 0x1.371a74p+0f, 0x1.3dea64p+0f, 0x1.44e086p+0f, 0x1.4bfdaep+0f, 0x1.5342b6p+0f, 0x1.5ab07ep+0f, 0x1.6247ecp+0f, 0x1.6a09e6p+0f, 0x1.71f75ep+0f, 0x1.7a1148p+0f, 0x1.82589ap+0f, 0x1.8ace54p+0f, 0x1.93737cp+0f, 0x1.9c4918p+0f, 0x1.a5503cp+0f, 0x1.ae89fap+0f, 0x1.b7f77p+0f, 0x1.c199bep+0f, 0x1.cb720ep+0f, 0x1.d5818ep+0f, 0x1.dfc974p+0f, 0x1.ea4afap+0f, 0x1.f50766p+0f};
__device__ const float GFP_EXP2F_T_L[32] = {0.0f, -0x1.9d4f52p-25f, 0x1.9f3122p-25f, -0x1.fdb496p-25f, -0x1.c15742p-27f, -0x1.d2e8cap-25f, 0x1.ceac48p-25f, -0x1.5c0424p-25f, 0x1.4636e2p-25f, -0x1.18aac6p-25f, 0x1.824684p-25f, 0x1.8624b4p-30f, -0x1.593abcp-25f, -0x1.2c561p-25f, -0x1.5bd5ecp-27f, -0x1.f8b55p-25f, 0x1.9fcef4p-26f, 0x1.1d8beep-25f, -0x1.829fdp-25f, -0x1.accc7cp-26f, 0x1.15506ep-27f, -0x1.e64744p-25f, 0x1.51f848p-27f, -0x1.b83b54p-25f, -0x1.a94b14p-26f, -0x1.a09438p-25f, -0x1.3d56b2p-27f, -0x1.8837ccp-27f, -0x1.822dbcp-27f, -0x1.908c94p-25f, 0x1.52486cp-27f, -0x1.246ebp-26f};
#define GFP_EXPF_INVLN2N_H 0x1.715476p+5f
#define GFP_EXPF_INVLN2N_L 0x1.4ae0cp-21f
#define GFP_EXPF_C0_H 0x1.c6af84p-20f
#define GFP_EXPF_C0_L 0x1.722472p-45f
#define GFP_EXPF_C1_H 0x1.ebfce6p-13f
#define GFP_EXPF_C1_L -0x1.e0a762p-38f
#define GFP_EXPF_C2_H 0x1.62e43p-6f
#define GFP_EXPF_C2_L -0x1.e75a54p-35f

// glibc's binary64 table split into float pairs: hi = float(d), lo = float(d - hi)
__device__ const float GFP_POWF_LOG2C_H[16] = {-0x1.efec66p-2f, -0x1.b0b684p-2f, -0x1.7418bp-2f, -0x1.39de92p-2f, -0x1.01d9cp-2f, -0x1.97c1d2p-3f, -0x1.2f9e3ap-3f, -0x1.960cbcp-4f, -0x1.a6f9dcp-5f, 0.0f, 0x1.338caap-4f, 0x1.476a96p-3f, 0x1.e840b4p-3f, 0x1.40646p-2f, 0x1.88e9c2p-2f, 0x1.ce0a44p-2f};
__device__ const float GFP_POWF_LOG2C_L[16] = {0x1.1a73fap-28f, 0x1.a5606cp-27f, -0x1.43f6fp-27f, 0x1.648c22p-28f, 0x1.81a93ap-27f, 0x1.312144p-29f, 0x1.8a186cp-28f, 0x1.0ee548p-33f, 0x1.371406p-30f, 0.0f, -0x1.b61586p-33f, -0x1.78edc8p-28f, 0x1.589c9ap-28f, -0x1.e7335cp-27f, 0x1.8374p-27f, 0x1.d62f7ap-27f};
#define GFP_POWF_A0_H 0x1.27616cp-2f
#define GFP_POWF_A0_L 0x1.292dc2p-27f
#define GFP_POWF_A1_H -0x1.71969ap-2f
#define GFP_POWF_A1_L -0x1.d719e8p-32f
#define GFP_POWF_A2_H 0x1.ec70a6p-2f
#define GFP_POWF_A2_L 0x1.94f75cp-27f
#define GFP_POWF_A3_H -0x1.715474p-1f
#define GFP_POWF_A3_L -0x1.17ded8p-26f
#define GFP_POWF_A4_H 0x1.715476p+0f
#define GFP_POWF_A4_L 0x1.4aae0ap-26f
#define GFP_EXP2F_C0_H 0x1.c6af84p-5f
#define GFP_EXP2F_C0_L 0x1.722472p-30f
#define GFP_EXP2F_C1_H 0x1.ebfce6p-3f
#define GFP_EXP2F_C1_L -0x1.e0a762p-28f
#define GFP_EXP2F_C2_H 0x1.62e43p-1f
#define GFP_EXP2F_C2_L -0x1.e75a54p-30f

// arwen_fastlibm_ff.h -- float-float arithmetic and the rounding test of the libm fast paths.

#define GLIBC_FASTPATH_UNSURE 0x7FBADBADu
// -DGLIBC_FASTPATH_DISABLE: every fast path answers UNSURE, so each copy runs
// its original binary64 code (tests compile both ways and compare).
#ifdef GLIBC_FASTPATH_DISABLE
#define GFP_DISABLED 1
#else
#define GFP_DISABLED 0
#endif

struct gfp_ff { float h, l; };

__device__ __forceinline__ gfp_ff gfp_ff_two_sum(float a, float b) {
    float s = __fadd_rn(a, b);
    float bb = __fsub_rn(s, a);
    return gfp_ff{s, __fadd_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fsub_rn(b, bb))};
}
__device__ __forceinline__ gfp_ff gfp_ff_fast_two_sum(float a, float b) {   // |a| >= |b| or a == 0
    float s = __fadd_rn(a, b);
    return gfp_ff{s, __fsub_rn(b, __fsub_rn(s, a))};
}
__device__ __forceinline__ gfp_ff gfp_ff_add(gfp_ff a, gfp_ff b) {
    gfp_ff s = gfp_ff_two_sum(a.h, b.h);
    gfp_ff t = gfp_ff_two_sum(a.l, b.l);
    s.l = __fadd_rn(s.l, t.h);
    s = gfp_ff_fast_two_sum(s.h, s.l);
    s.l = __fadd_rn(s.l, t.l);
    return gfp_ff_fast_two_sum(s.h, s.l);
}
__device__ __forceinline__ gfp_ff gfp_ff_mul(gfp_ff a, gfp_ff b) {
    float p = __fmul_rn(a.h, b.h);
    float e = __fmaf_rn(a.h, b.h, -p);
    e = __fmaf_rn(a.h, b.l, e);
    e = __fmaf_rn(a.l, b.h, e);
    return gfp_ff_fast_two_sum(p, e);
}
__device__ __forceinline__ gfp_ff gfp_ff_of(float h, float l) { return gfp_ff{h, l}; }

// The binary32 rounding of any value within E of h + l, if it is certainly h:
// the interval [h + l - E, h + l + E] lies strictly between the rounding
// boundaries around h, h + ulp/2 above and h - ulp/2 below -- or h - ulp/4
// below when h is a power of two (the grid below is twice as fine).
__device__ __forceinline__ float gfp_ff_round_sure(gfp_ff y, float E) {
    unsigned int hb = __float_as_uint(y.h);
    unsigned int e = hb & 0x7f800000u;
    // tiny (keeps the half-ulp computation normal), inf or nan
    if (e <= (26u << 23) || e == 0x7f800000u)
        return __uint_as_float(GLIBC_FASTPATH_UNSURE);
    float half_ulp = __uint_as_float(e - (24u << 23));
    float below = ((hb & 0x007fffffu) == 0u) ? 0.5f * half_ulp : half_ulp;
    if (y.l + E >= half_ulp || E - y.l >= below)
        return __uint_as_float(GLIBC_FASTPATH_UNSURE);
    return y.h;
}


// arwen_fastlibm_logf.h -- fast path of the glibc logf copies (see arwen_fastlibm.h).

// glibc logf: y = (k ln2 + logc) + r + r^2 (A1 r + A2 + A0 r^2), r = z invc - 1.
__device__ __forceinline__ float glibc_fastpath_logf(float x) {
    if (GFP_DISABLED) return __uint_as_float(GLIBC_FASTPATH_UNSURE);
    unsigned int ix = __float_as_uint(x);
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u || ix == 0x3f800000u)
        return __uint_as_float(GLIBC_FASTPATH_UNSURE);       // 0, subnormal, negative, inf, nan, 1
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    int k = ((int)tmp) >> 23;
    float z = __uint_as_float(ix - (tmp & 0xff800000u));
    // r = z * invc - 1: z * invc_h exactly as p + pe, p - 1 exact (Sterbenz)
    float p = __fmul_rn(z, GFP_LOGF_INVC_H[i]);
    float pe = __fmaf_rn(z, GFP_LOGF_INVC_H[i], -p);
    pe = __fmaf_rn(z, GFP_LOGF_INVC_L[i], pe);
    gfp_ff r = gfp_ff_two_sum(p - 1.0f, pe);
    // y0 = k ln2 + logc (k * GFP_LN2_1, k * GFP_LN2_2 exact for |k| < 256)
    float kf = (float)k;
    gfp_ff y0 = gfp_ff_two_sum(kf * GFP_LN2_1, GFP_LOGF_LOGC_H[i]);
    gfp_ff y1 = gfp_ff_two_sum(kf * GFP_LN2_2, GFP_LOGF_LOGC_L[i]);
    y1.l = __fmaf_rn(kf, GFP_LN2_3, y1.l);
    y0 = gfp_ff_add(y0, y1);
    gfp_ff r2 = gfp_ff_mul(r, r);
    gfp_ff q = gfp_ff_add(gfp_ff_mul(gfp_ff_of(GFP_LOGF_A1_H, GFP_LOGF_A1_L), r),
                                  gfp_ff_of(GFP_LOGF_A2_H, GFP_LOGF_A2_L));
    q = gfp_ff_add(q, gfp_ff_mul(gfp_ff_of(GFP_LOGF_A0_H, GFP_LOGF_A0_L), r2));
    q = gfp_ff_mul(q, r2);
    gfp_ff y = gfp_ff_add(gfp_ff_add(y0, r), q);
    // budget: float-float evaluation and binary64 rounding of the copies,
    // relative to the summands (cancellation), plus z*invc rounded to binary64
    // in the mul+sub copies (2^-53 absolute)
    float E = 0x1p-36f * (fabsf(y0.h) + fabsf(r.h)) + 0x1p-50f;
    return gfp_ff_round_sure(y, E);
}


// arwen_fastlibm_expf.h -- fast path of the glibc expf copies (see arwen_fastlibm.h).

// glibc expf: k = round(x 32/ln2), r = x 32/ln2 - k,
// y = 2^(k/32) (1 + C2 r + (C1 + C0 r) r^2), 2^(k/32) = T[k & 31] 2^(k >> 5).
__device__ __forceinline__ float glibc_fastpath_expf(float x) {
    if (GFP_DISABLED) return __uint_as_float(GLIBC_FASTPATH_UNSURE);
    unsigned int ix = __float_as_uint(x);
    unsigned int ax = ix & 0x7fffffffu;
    // |x| >= 80 (result near overflow / underflow), inf, nan, subnormal: unsure
    if (ax >= 0x42a00000u || (ax != 0u && ax < 0x00800000u))
        return __uint_as_float(GLIBC_FASTPATH_UNSURE);
    // t = x * InvLn2N as float-float (exact product with the high part)
    float th = __fmul_rn(x, GFP_EXPF_INVLN2N_H);
    float tl = __fmaf_rn(x, GFP_EXPF_INVLN2N_H, -th);
    tl = __fmaf_rn(x, GFP_EXPF_INVLN2N_L, tl);
    float kf = rintf(th);
    float f = (th - kf) + tl;                      // th - kf exact (|th| < 2^13)
    // the copies round x * InvLn2N (fma or rounded product) to the nearest
    // integer: undecided within 2^-30 of a half
    float af = fabsf(f);
    if (fabsf(af - 0.5f) < 0x1p-30f)
        return __uint_as_float(GLIBC_FASTPATH_UNSURE);
    if (f > 0.5f) kf += 1.0f;
    if (f < -0.5f) kf -= 1.0f;
    gfp_ff r = gfp_ff_two_sum(th - kf, tl);
    int k = (int)kf;
    int j = k & 31;
    // P(r) = 1 + C2 r + (C1 + C0 r) r^2
    gfp_ff c1 = gfp_ff_add(gfp_ff_of(GFP_EXPF_C1_H, GFP_EXPF_C1_L),
                                   gfp_ff_mul(gfp_ff_of(GFP_EXPF_C0_H, GFP_EXPF_C0_L), r));
    gfp_ff pr = gfp_ff_add(gfp_ff_mul(gfp_ff_of(GFP_EXPF_C2_H, GFP_EXPF_C2_L), r),
                                   gfp_ff_mul(c1, gfp_ff_mul(r, r)));
    pr = gfp_ff_add(gfp_ff_of(1.0f, 0.0f), pr);
    gfp_ff y = gfp_ff_mul(pr, gfp_ff_of(GFP_EXP2F_T_H[j], GFP_EXP2F_T_L[j]));
    // decide on the unscaled value in [~0.98, ~2.03]; the scale 2^(k >> 5) is exact
    // for the normal results |x| < 80 gives
    float E = 0x1p-36f * fabsf(y.h);
    float m = gfp_ff_round_sure(y, E);
    if (__float_as_uint(m) == GLIBC_FASTPATH_UNSURE) return m;
    return m * __uint_as_float((unsigned int)(((k >> 5) + 127) << 23));
}

// arwen_fastlibm_powf.h -- fast path of the glibc powf copies (see arwen_fastlibm.h).
//
// glibc powf for x > 0: log2(x) = (log2 c + k) + A4 r + A2 r^2 ... (table and
// degree-5 polynomial, r = z invc - 1), ylogx = y log2(x), then
// 2^ylogx = T[k & 31] 2^(k >> 5) (1 + C2 r + (C0 r + C1) r^2), k = round(32 ylogx),
// r = ylogx - k/32.  Negative or special x, special y, and |ylogx| >= 120 are
// left to the exact original.  gpuwm.metal.fastlibm routes only call sites
// whose exponent is a literal, each (copy, exponent) pair verified on all 2^32 x.

__device__ __forceinline__ float glibc_fastpath_powf(float x, float y) {
    if (GFP_DISABLED) return __uint_as_float(GLIBC_FASTPATH_UNSURE);
    unsigned int ix = __float_as_uint(x);
    unsigned int ay = __float_as_uint(y) & 0x7fffffffu;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u || ix == 0x3f800000u ||
        ay - 0x00800000u >= 0x7f800000u - 0x00800000u)
        return __uint_as_float(GLIBC_FASTPATH_UNSURE);
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    unsigned int top = tmp & 0xff800000u;
    float z = __uint_as_float(ix - top);
    float kf = (float)(((int)top) >> 23);
    // r = z invc - 1 (as logf)
    float p = __fmul_rn(z, GFP_LOGF_INVC_H[i]);
    float pe = __fmaf_rn(z, GFP_LOGF_INVC_H[i], -p);
    pe = __fmaf_rn(z, GFP_LOGF_INVC_L[i], pe);
    gfp_ff r = gfp_ff_two_sum(p - 1.0f, pe);
    gfp_ff y0 = gfp_ff_add(gfp_ff_of(kf, 0.0f), gfp_ff_of(GFP_POWF_LOG2C_H[i], GFP_POWF_LOG2C_L[i]));
    gfp_ff r2 = gfp_ff_mul(r, r);
    gfp_ff r4 = gfp_ff_mul(r2, r2);
    gfp_ff a = gfp_ff_add(gfp_ff_mul(gfp_ff_of(GFP_POWF_A0_H, GFP_POWF_A0_L), r),
                                  gfp_ff_of(GFP_POWF_A1_H, GFP_POWF_A1_L));
    gfp_ff b = gfp_ff_add(gfp_ff_mul(gfp_ff_of(GFP_POWF_A2_H, GFP_POWF_A2_L), r),
                                  gfp_ff_of(GFP_POWF_A3_H, GFP_POWF_A3_L));
    gfp_ff q = gfp_ff_add(gfp_ff_mul(gfp_ff_of(GFP_POWF_A4_H, GFP_POWF_A4_L), r), y0);
    q = gfp_ff_add(gfp_ff_mul(b, r2), q);
    gfp_ff logx = gfp_ff_add(gfp_ff_mul(a, r4), q);
    gfp_ff ylogx = gfp_ff_mul(gfp_ff_of(y, 0.0f), logx);
    if (!(fabsf(ylogx.h) < 120.0f))
        return __uint_as_float(GLIBC_FASTPATH_UNSURE);
    // k = round(32 ylogx): the copies round a binary64 32*ylogx; undecided near a half
    float th = ylogx.h * 32.0f, tl = ylogx.l * 32.0f;
    float k2 = rintf(th);
    float f = (th - k2) + tl;
    if (fabsf(fabsf(f) - 0.5f) < 0x1p-26f)
        return __uint_as_float(GLIBC_FASTPATH_UNSURE);
    if (f > 0.5f) k2 += 1.0f;
    if (f < -0.5f) k2 -= 1.0f;
    gfp_ff rr = gfp_ff_two_sum(th - k2, tl);
    rr.h *= 0x1p-5f;
    rr.l *= 0x1p-5f;
    int kk = (int)k2;
    int j = kk & 31;
    gfp_ff zc = gfp_ff_add(gfp_ff_mul(gfp_ff_of(GFP_EXP2F_C0_H, GFP_EXP2F_C0_L), rr),
                                   gfp_ff_of(GFP_EXP2F_C1_H, GFP_EXP2F_C1_L));
    gfp_ff pr = gfp_ff_add(gfp_ff_mul(gfp_ff_of(GFP_EXP2F_C2_H, GFP_EXP2F_C2_L), rr),
                                   gfp_ff_mul(zc, gfp_ff_mul(rr, rr)));
    pr = gfp_ff_add(gfp_ff_of(1.0f, 0.0f), pr);
    gfp_ff res = gfp_ff_mul(pr, gfp_ff_of(GFP_EXP2F_T_H[j], GFP_EXP2F_T_L[j]));
    // budget: binary64 roundings of the copies and the float-float evaluation,
    // the log2 error scaled by |y| (relative to the summands: cancellation near
    // x = c) and amplified into 2^ylogx
    float E = fabsf(res.h) * (0x1p-38f + 0x1p-40f * (fabsf(ylogx.h) +
                                    fabsf(y) * (fabsf(y0.h) + 2.0f * fabsf(r.h))));
    float m = gfp_ff_round_sure(res, E);
    if (__float_as_uint(m) == GLIBC_FASTPATH_UNSURE) return m;
    return m * __uint_as_float((unsigned int)(((kk >> 5) + 127) << 23));
}
#endif  // GLIBC_FASTPATH_CUH
