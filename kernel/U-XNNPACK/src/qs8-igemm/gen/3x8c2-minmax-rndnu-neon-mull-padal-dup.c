// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c2-neon-mull-padal-dup.c.in
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


void xnn_qs8_igemm_minmax_rndnu_ukernel_3x8c2__neon_mull_padal_dup(
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
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 2 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      a += 3;

      size_t k = kc;


      while (k >= 8 * sizeof(int8_t)) {
        const int8x8_t va0 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1 = vld1_s8(a1); a1 += 8;
        const int8x8_t va2 = vld1_s8(a2); a2 += 8;

        const int8x8_t vb0123c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb0123c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb0123c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb0123c3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int8x8_t va0c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0));
        const int8x8_t va1c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0));
        const int8x8_t va2c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 0));
        const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, va0c0);
        const int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0, va1c0);
        const int16x8_t vprod2x0123c0 = vmull_s8(vb0123c0, va2c0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c0);
        const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, va0c0);
        const int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0, va1c0);
        const int16x8_t vprod2x4567c0 = vmull_s8(vb4567c0, va2c0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c0);
        const int8x8_t va0c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1));
        const int8x8_t va1c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1));
        const int8x8_t va2c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 1));
        const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, va0c1);
        const int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1, va1c1);
        const int16x8_t vprod2x0123c1 = vmull_s8(vb0123c1, va2c1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c1);
        const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, va0c1);
        const int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1, va1c1);
        const int16x8_t vprod2x4567c1 = vmull_s8(vb4567c1, va2c1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c1);
        const int8x8_t va0c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2));
        const int8x8_t va1c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2));
        const int8x8_t va2c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 2));
        const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, va0c2);
        const int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2, va1c2);
        const int16x8_t vprod2x0123c2 = vmull_s8(vb0123c2, va2c2);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c2);
        const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, va0c2);
        const int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2, va1c2);
        const int16x8_t vprod2x4567c2 = vmull_s8(vb4567c2, va2c2);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c2);
        const int8x8_t va0c3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3));
        const int8x8_t va1c3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 3));
        const int8x8_t va2c3 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 3));
        const int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3, va0c3);
        const int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3, va1c3);
        const int16x8_t vprod2x0123c3 = vmull_s8(vb0123c3, va2c3);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c3);
        const int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3, va0c3);
        const int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3, va1c3);
        const int16x8_t vprod2x4567c3 = vmull_s8(vb4567c3, va2c3);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c3);

        k -= 8 * sizeof(int8_t);
      }

      if XNN_UNLIKELY(k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
        const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);
        const int8x8_t va2 = vld1_s8(a2); a2 = (const int8_t*) ((uintptr_t) a2 + k);

        const int8x8_t vb0123c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int8x8_t va0c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0));
        const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, va0c0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, va0c0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        const int8x8_t va1c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0));
        const int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0, va1c0);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
        const int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0, va1c0);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
        const int8x8_t va2c0 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 0));
        const int16x8_t vprod2x0123c0 = vmull_s8(vb0123c0, va2c0);
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c0);
        const int16x8_t vprod2x4567c0 = vmull_s8(vb4567c0, va2c0);
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c0);

        if (k > 2 * sizeof(int8_t)) {
          const int8x8_t vb0123c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vb4567c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

          const int8x8_t va0c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1));
          const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, va0c1);
          vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
          const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, va0c1);
          vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
          const int8x8_t va1c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1));
          const int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1, va1c1);
          vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
          const int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1, va1c1);
          vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
          const int8x8_t va2c1 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 1));
          const int16x8_t vprod2x0123c1 = vmull_s8(vb0123c1, va2c1);
          vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c1);
          const int16x8_t vprod2x4567c1 = vmull_s8(vb4567c1, va2c1);
          vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c1);

          if (k > 4 * sizeof(int8_t)) {
            const int8x8_t vb0123c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
            const int8x8_t vb4567c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

            const int8x8_t va0c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2));
            const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, va0c2);
            vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
            const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, va0c2);
            vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
            const int8x8_t va1c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2));
            const int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2, va1c2);
            vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
            const int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2, va1c2);
            vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
            const int8x8_t va2c2 = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 2));
            const int16x8_t vprod2x0123c2 = vmull_s8(vb0123c2, va2c2);
            vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c2);
            const int16x8_t vprod2x4567c2 = vmull_s8(vb4567c2, va2c2);
            vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c2);
          }
        }
      }
      p -= 3 * sizeof(void*);
    } while (p != 0);

    const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
    const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);

    vacc0x0123 = vshlq_s32(vacc0x0123, vright_pre_shift);
    vacc0x4567 = vshlq_s32(vacc0x4567, vright_pre_shift);
    vacc1x0123 = vshlq_s32(vacc1x0123, vright_pre_shift);
    vacc1x4567 = vshlq_s32(vacc1x4567, vright_pre_shift);
    vacc2x0123 = vshlq_s32(vacc2x0123, vright_pre_shift);
    vacc2x4567 = vshlq_s32(vacc2x4567, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);
    vacc1x0123 = vqdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqdmulhq_s32(vacc1x4567, vmultiplier);
    vacc2x0123 = vqdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqdmulhq_s32(vacc2x4567, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_post_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_post_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_post_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
    int8x8_t vout2x01234567 = vqmovn_s16(vacc2x01234567);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
    int8x8_t vout2x01234567 = vqmovn_s16(vacc2x01234567);
#endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->rndnu_neon.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->rndnu_neon.output_max);

    vout2x01234567 = vmax_s8(vout2x01234567, vget_low_s8(voutput_min));
    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);

    vout2x01234567 = vmin_s8(vout2x01234567, vget_low_s8(voutput_max));
    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c2 + 0, vout2x01234567);
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));

      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpret_u32_s8(vout2x01234567), 0); c2 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vout2x01234567 = vext_s8(vout2x01234567, vout2x01234567, 4);
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpret_u16_s8(vout2x01234567), 0); c2 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vout2x01234567 = vext_s8(vout2x01234567, vout2x01234567, 2);
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c2, vout2x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
