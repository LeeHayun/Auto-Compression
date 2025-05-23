// Auto-generated file. Do not edit!
//   Template: src/f16-igemm/neonfp16arith-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/igemm.h>


void xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const void** restrict a,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const void* zero,
    const struct xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(__fp16) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(__fp16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  __fp16* c0 = (__fp16*) c;

  do {
    float16x8_t vacc0x01234567 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));
    float16x8_t vacc0x89ABCDEF = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

    size_t p = ks;
    do {
      const __fp16* restrict a0 = (const __fp16*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const __fp16*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      for (; k >= 4 * sizeof(__fp16); k -= 4 * sizeof(__fp16)) {
        const float16x4_t va0 = vld1_f16(a0); a0 += 4;

        const float16x8_t vb01234567c0 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));
        const float16x8_t vb89ABCDEFc0 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));

        #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c0, va0, 0);
          vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc0, va0, 0);
        #else
          const float16x8_t va0c0 = vdupq_lane_f16(va0, 0);

          vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0c0, vb01234567c0);
          vacc0x89ABCDEF = vfmaq_f16(vacc0x89ABCDEF, va0c0, vb89ABCDEFc0);
        #endif
        const float16x8_t vb01234567c1 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));
        const float16x8_t vb89ABCDEFc1 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));

        #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c1, va0, 1);
          vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc1, va0, 1);
        #else
          const float16x8_t va0c1 = vdupq_lane_f16(va0, 1);

          vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0c1, vb01234567c1);
          vacc0x89ABCDEF = vfmaq_f16(vacc0x89ABCDEF, va0c1, vb89ABCDEFc1);
        #endif
        const float16x8_t vb01234567c2 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));
        const float16x8_t vb89ABCDEFc2 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));

        #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c2, va0, 2);
          vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc2, va0, 2);
        #else
          const float16x8_t va0c2 = vdupq_lane_f16(va0, 2);

          vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0c2, vb01234567c2);
          vacc0x89ABCDEF = vfmaq_f16(vacc0x89ABCDEF, va0c2, vb89ABCDEFc2);
        #endif
        const float16x8_t vb01234567c3 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));
        const float16x8_t vb89ABCDEFc3 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof( float16x8_t));

        #if XNN_ARCH_ARM64
          vacc0x01234567 = vfmaq_lane_f16(vacc0x01234567, vb01234567c3, va0, 3);
          vacc0x89ABCDEF = vfmaq_lane_f16(vacc0x89ABCDEF, vb89ABCDEFc3, va0, 3);
        #else
          const float16x8_t va0c3 = vdupq_lane_f16(va0, 3);

          vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0c3, vb01234567c3);
          vacc0x89ABCDEF = vfmaq_f16(vacc0x89ABCDEF, va0c3, vb89ABCDEFc3);
        #endif
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const float16x8_t va0 = vld1q_dup_f16(a0); a0 += 1;

          const float16x8_t vb01234567 = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));
          const float16x8_t vb89ABCDEF = vld1q_f16(w); w = (const void*) ((uintptr_t) w + sizeof(float16x8_t));

          vacc0x01234567 = vfmaq_f16(vacc0x01234567, va0, vb01234567);
          vacc0x89ABCDEF = vfmaq_f16(vacc0x89ABCDEF, va0, vb89ABCDEF);

          k -= sizeof(__fp16);
        } while (k != 0);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->scale));
    vacc0x01234567 = vmulq_f16(vacc0x01234567, vscale);
    vacc0x89ABCDEF = vmulq_f16(vacc0x89ABCDEF, vscale);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->max));
    vacc0x01234567 = vminq_f16(vacc0x01234567, vmax);
    vacc0x89ABCDEF = vminq_f16(vacc0x89ABCDEF, vmax);

    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->min));
    vacc0x01234567 = vmaxq_f16(vacc0x01234567, vmin);
    vacc0x89ABCDEF = vmaxq_f16(vacc0x89ABCDEF, vmin);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f16(c0, vacc0x01234567);
      vst1q_f16(c0 + 8, vacc0x89ABCDEF);
      c0 = (__fp16*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_f16(c0, vacc0x01234567); c0 += 8;

        vacc0x01234567 = vacc0x89ABCDEF;
      }
      float16x4_t vacc0x0123 = vget_low_f16(vacc0x01234567);
      if (nc & 4) {
        vst1_f16(c0, vacc0x0123); c0 += 4;

        vacc0x0123 = vget_high_f16(vacc0x01234567);
      }
      if (nc & 2) {
        vst1_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpret_u32_f16(vacc0x0123), 0); c0 += 2;

        vacc0x0123 = vext_f16(vacc0x0123, vacc0x0123, 2);
      }
      if (nc & 1) {
        vst1_lane_f16(c0, vacc0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
