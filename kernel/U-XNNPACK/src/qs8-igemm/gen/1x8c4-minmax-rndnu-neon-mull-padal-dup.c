// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c4-neon-mull-padal-dup.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>


void xnn_qs8_igemm_minmax_rndnu_ukernel_1x8c4__neon_mull_padal_dup(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  int8_t* c0 = c;

  do {
    int32x4_t vacc0x01 = vreinterpretq_s32_u64(vshll_n_u32(vld1_u32(w), 0)); w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));
    int32x4_t vacc0x23 = vreinterpretq_s32_u64(vshll_n_u32(vld1_u32(w), 0)); w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));
    int32x4_t vacc0x45 = vreinterpretq_s32_u64(vshll_n_u32(vld1_u32(w), 0)); w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));
    int32x4_t vacc0x67 = vreinterpretq_s32_u64(vshll_n_u32(vld1_u32(w), 0)); w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t));

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;


      while (k >= 8 * sizeof(int8_t)) {
        const int8x8_t va0 = vld1_s8(a0); a0 += 8;

        const int8x8_t vb01c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb23c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb45c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb67c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb01c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb23c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb45c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb67c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int8x8_t va0c0 = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(va0), 0));
        const int16x8_t vprod0x01c0 = vmull_s8(vb01c0, va0c0);
        vacc0x01 = vpadalq_s16(vacc0x01, vprod0x01c0);
        const int16x8_t vprod0x23c0 = vmull_s8(vb23c0, va0c0);
        vacc0x23 = vpadalq_s16(vacc0x23, vprod0x23c0);
        const int16x8_t vprod0x45c0 = vmull_s8(vb45c0, va0c0);
        vacc0x45 = vpadalq_s16(vacc0x45, vprod0x45c0);
        const int16x8_t vprod0x67c0 = vmull_s8(vb67c0, va0c0);
        vacc0x67 = vpadalq_s16(vacc0x67, vprod0x67c0);
        const int8x8_t va0c1 = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(va0), 1));
        const int16x8_t vprod0x01c1 = vmull_s8(vb01c1, va0c1);
        vacc0x01 = vpadalq_s16(vacc0x01, vprod0x01c1);
        const int16x8_t vprod0x23c1 = vmull_s8(vb23c1, va0c1);
        vacc0x23 = vpadalq_s16(vacc0x23, vprod0x23c1);
        const int16x8_t vprod0x45c1 = vmull_s8(vb45c1, va0c1);
        vacc0x45 = vpadalq_s16(vacc0x45, vprod0x45c1);
        const int16x8_t vprod0x67c1 = vmull_s8(vb67c1, va0c1);
        vacc0x67 = vpadalq_s16(vacc0x67, vprod0x67c1);

        k -= 8 * sizeof(int8_t);
      }

      if XNN_UNLIKELY(k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);

        const int8x8_t vb01c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb23c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb45c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb67c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int8x8_t va0c0 = vreinterpret_s8_s32(vdup_lane_s32(vreinterpret_s32_s8(va0), 0));
        const int16x8_t vprod0x01c0 = vmull_s8(vb01c0, va0c0);
        vacc0x01 = vpadalq_s16(vacc0x01, vprod0x01c0);
        const int16x8_t vprod0x23c0 = vmull_s8(vb23c0, va0c0);
        vacc0x23 = vpadalq_s16(vacc0x23, vprod0x23c0);
        const int16x8_t vprod0x45c0 = vmull_s8(vb45c0, va0c0);
        vacc0x45 = vpadalq_s16(vacc0x45, vprod0x45c0);
        const int16x8_t vprod0x67c0 = vmull_s8(vb67c0, va0c0);
        vacc0x67 = vpadalq_s16(vacc0x67, vprod0x67c0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

#if XNN_ARCH_ARM64
    int32x4_t vacc0x0123 = vpaddq_s32(vacc0x01, vacc0x23);
    int32x4_t vacc0x4567 = vpaddq_s32(vacc0x45, vacc0x67);
#else
    const int32x2_t vsum0x01 = vpadd_s32(vget_low_s32(vacc0x01), vget_high_s32(vacc0x01));
    const int32x2_t vsum0x23 = vpadd_s32(vget_low_s32(vacc0x23), vget_high_s32(vacc0x23));
    int32x4_t vacc0x0123 = vcombine_s32(vsum0x01, vsum0x23);
    const int32x2_t vsum0x45 = vpadd_s32(vget_low_s32(vacc0x45), vget_high_s32(vacc0x45));
    const int32x2_t vsum0x67 = vpadd_s32(vget_low_s32(vacc0x67), vget_high_s32(vacc0x67));
    int32x4_t vacc0x4567 = vcombine_s32(vsum0x45, vsum0x67);
#endif

    const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
    const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);

    vacc0x0123 = vshlq_s32(vacc0x0123, vright_pre_shift);
    vacc0x4567 = vshlq_s32(vacc0x4567, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);

    int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);

    int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
#endif
    const int8x8_t voutput_min = vld1_dup_s8(&params->rndnu_neon.output_min);
    const int8x8_t voutput_max = vld1_dup_s8(&params->rndnu_neon.output_max);

    vout0x01234567 = vmax_s8(vout0x01234567, voutput_min);

    vout0x01234567 = vmin_s8(vout0x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c0 + 0, vout0x01234567);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
