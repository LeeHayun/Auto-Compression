// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/neon-mull-addw-dup.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/gemm.h>


void xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__neon_mull_addw_dup(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;

    size_t k = kc;
    while (k >= 8 * sizeof(int8_t)) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3 = vld1_s8(a3); a3 += 8;

      const int8x8_t vb01234567c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x01234567c0 = vmull_s8(vb01234567c0, vdup_lane_s8(va0, 0));
      vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c0));
      vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c0));
      const int16x8_t vprod1x01234567c0 = vmull_s8(vb01234567c0, vdup_lane_s8(va1, 0));
      vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c0));
      vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c0));
      const int16x8_t vprod2x01234567c0 = vmull_s8(vb01234567c0, vdup_lane_s8(va2, 0));
      vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c0));
      vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c0));
      const int16x8_t vprod3x01234567c0 = vmull_s8(vb01234567c0, vdup_lane_s8(va3, 0));
      vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c0));
      vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c0));
      const int8x8_t vb01234567c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x01234567c1 = vmull_s8(vb01234567c1, vdup_lane_s8(va0, 1));
      vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c1));
      vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c1));
      const int16x8_t vprod1x01234567c1 = vmull_s8(vb01234567c1, vdup_lane_s8(va1, 1));
      vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c1));
      vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c1));
      const int16x8_t vprod2x01234567c1 = vmull_s8(vb01234567c1, vdup_lane_s8(va2, 1));
      vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c1));
      vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c1));
      const int16x8_t vprod3x01234567c1 = vmull_s8(vb01234567c1, vdup_lane_s8(va3, 1));
      vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c1));
      vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c1));
      const int8x8_t vb01234567c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x01234567c2 = vmull_s8(vb01234567c2, vdup_lane_s8(va0, 2));
      vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c2));
      vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c2));
      const int16x8_t vprod1x01234567c2 = vmull_s8(vb01234567c2, vdup_lane_s8(va1, 2));
      vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c2));
      vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c2));
      const int16x8_t vprod2x01234567c2 = vmull_s8(vb01234567c2, vdup_lane_s8(va2, 2));
      vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c2));
      vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c2));
      const int16x8_t vprod3x01234567c2 = vmull_s8(vb01234567c2, vdup_lane_s8(va3, 2));
      vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c2));
      vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c2));
      const int8x8_t vb01234567c3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x01234567c3 = vmull_s8(vb01234567c3, vdup_lane_s8(va0, 3));
      vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c3));
      vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c3));
      const int16x8_t vprod1x01234567c3 = vmull_s8(vb01234567c3, vdup_lane_s8(va1, 3));
      vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c3));
      vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c3));
      const int16x8_t vprod2x01234567c3 = vmull_s8(vb01234567c3, vdup_lane_s8(va2, 3));
      vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c3));
      vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c3));
      const int16x8_t vprod3x01234567c3 = vmull_s8(vb01234567c3, vdup_lane_s8(va3, 3));
      vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c3));
      vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c3));
      const int8x8_t vb01234567c4 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x01234567c4 = vmull_s8(vb01234567c4, vdup_lane_s8(va0, 4));
      vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c4));
      vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c4));
      const int16x8_t vprod1x01234567c4 = vmull_s8(vb01234567c4, vdup_lane_s8(va1, 4));
      vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c4));
      vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c4));
      const int16x8_t vprod2x01234567c4 = vmull_s8(vb01234567c4, vdup_lane_s8(va2, 4));
      vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c4));
      vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c4));
      const int16x8_t vprod3x01234567c4 = vmull_s8(vb01234567c4, vdup_lane_s8(va3, 4));
      vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c4));
      vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c4));
      const int8x8_t vb01234567c5 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x01234567c5 = vmull_s8(vb01234567c5, vdup_lane_s8(va0, 5));
      vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c5));
      vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c5));
      const int16x8_t vprod1x01234567c5 = vmull_s8(vb01234567c5, vdup_lane_s8(va1, 5));
      vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c5));
      vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c5));
      const int16x8_t vprod2x01234567c5 = vmull_s8(vb01234567c5, vdup_lane_s8(va2, 5));
      vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c5));
      vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c5));
      const int16x8_t vprod3x01234567c5 = vmull_s8(vb01234567c5, vdup_lane_s8(va3, 5));
      vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c5));
      vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c5));
      const int8x8_t vb01234567c6 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x01234567c6 = vmull_s8(vb01234567c6, vdup_lane_s8(va0, 6));
      vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c6));
      vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c6));
      const int16x8_t vprod1x01234567c6 = vmull_s8(vb01234567c6, vdup_lane_s8(va1, 6));
      vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c6));
      vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c6));
      const int16x8_t vprod2x01234567c6 = vmull_s8(vb01234567c6, vdup_lane_s8(va2, 6));
      vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c6));
      vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c6));
      const int16x8_t vprod3x01234567c6 = vmull_s8(vb01234567c6, vdup_lane_s8(va3, 6));
      vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c6));
      vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c6));
      const int8x8_t vb01234567c7 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x01234567c7 = vmull_s8(vb01234567c7, vdup_lane_s8(va0, 7));
      vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c7));
      vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c7));
      const int16x8_t vprod1x01234567c7 = vmull_s8(vb01234567c7, vdup_lane_s8(va1, 7));
      vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c7));
      vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c7));
      const int16x8_t vprod2x01234567c7 = vmull_s8(vb01234567c7, vdup_lane_s8(va2, 7));
      vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c7));
      vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c7));
      const int16x8_t vprod3x01234567c7 = vmull_s8(vb01234567c7, vdup_lane_s8(va3, 7));
      vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c7));
      vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c7));

      k -= 8 * sizeof(int8_t);
    }
    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
      const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);
      const int8x8_t va2 = vld1_s8(a2); a2 = (const int8_t*) ((uintptr_t) a2 + k);
      const int8x8_t va3 = vld1_s8(a3); a3 = (const int8_t*) ((uintptr_t) a3 + k);

      const int8x8_t vb01234567c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x01234567c0 = vmull_s8(vb01234567c0, vdup_lane_s8(va0, 0));
      vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c0));
      vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c0));
      const int16x8_t vprod1x01234567c0 = vmull_s8(vb01234567c0, vdup_lane_s8(va1, 0));
      vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c0));
      vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c0));
      const int16x8_t vprod2x01234567c0 = vmull_s8(vb01234567c0, vdup_lane_s8(va2, 0));
      vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c0));
      vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c0));
      const int16x8_t vprod3x01234567c0 = vmull_s8(vb01234567c0, vdup_lane_s8(va3, 0));
      vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c0));
      vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c0));

      if (k >= 2 * sizeof(int8_t)) {
        const int8x8_t vb01234567c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x01234567c1 = vmull_s8(vb01234567c1, vdup_lane_s8(va0, 1));
        vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c1));
        vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c1));
        const int16x8_t vprod1x01234567c1 = vmull_s8(vb01234567c1, vdup_lane_s8(va1, 1));
        vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c1));
        vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c1));
        const int16x8_t vprod2x01234567c1 = vmull_s8(vb01234567c1, vdup_lane_s8(va2, 1));
        vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c1));
        vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c1));
        const int16x8_t vprod3x01234567c1 = vmull_s8(vb01234567c1, vdup_lane_s8(va3, 1));
        vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c1));
        vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c1));

        if (k > 2 * sizeof(int8_t)) {
          const int8x8_t vb01234567c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

          const int16x8_t vprod0x01234567c2 = vmull_s8(vb01234567c2, vdup_lane_s8(va0, 2));
          vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c2));
          vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c2));
          const int16x8_t vprod1x01234567c2 = vmull_s8(vb01234567c2, vdup_lane_s8(va1, 2));
          vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c2));
          vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c2));
          const int16x8_t vprod2x01234567c2 = vmull_s8(vb01234567c2, vdup_lane_s8(va2, 2));
          vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c2));
          vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c2));
          const int16x8_t vprod3x01234567c2 = vmull_s8(vb01234567c2, vdup_lane_s8(va3, 2));
          vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c2));
          vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c2));

          if (k >= 4 * sizeof(int8_t)) {
            const int8x8_t vb01234567c3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

            const int16x8_t vprod0x01234567c3 = vmull_s8(vb01234567c3, vdup_lane_s8(va0, 3));
            vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c3));
            vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c3));
            const int16x8_t vprod1x01234567c3 = vmull_s8(vb01234567c3, vdup_lane_s8(va1, 3));
            vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c3));
            vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c3));
            const int16x8_t vprod2x01234567c3 = vmull_s8(vb01234567c3, vdup_lane_s8(va2, 3));
            vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c3));
            vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c3));
            const int16x8_t vprod3x01234567c3 = vmull_s8(vb01234567c3, vdup_lane_s8(va3, 3));
            vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c3));
            vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c3));

            if (k > 4 * sizeof(int8_t)) {
              const int8x8_t vb01234567c4 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

              const int16x8_t vprod0x01234567c4 = vmull_s8(vb01234567c4, vdup_lane_s8(va0, 4));
              vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c4));
              vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c4));
              const int16x8_t vprod1x01234567c4 = vmull_s8(vb01234567c4, vdup_lane_s8(va1, 4));
              vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c4));
              vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c4));
              const int16x8_t vprod2x01234567c4 = vmull_s8(vb01234567c4, vdup_lane_s8(va2, 4));
              vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c4));
              vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c4));
              const int16x8_t vprod3x01234567c4 = vmull_s8(vb01234567c4, vdup_lane_s8(va3, 4));
              vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c4));
              vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c4));

              if (k >= 6 * sizeof(int8_t)) {
                const int8x8_t vb01234567c5 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

                const int16x8_t vprod0x01234567c5 = vmull_s8(vb01234567c5, vdup_lane_s8(va0, 5));
                vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c5));
                vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c5));
                const int16x8_t vprod1x01234567c5 = vmull_s8(vb01234567c5, vdup_lane_s8(va1, 5));
                vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c5));
                vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c5));
                const int16x8_t vprod2x01234567c5 = vmull_s8(vb01234567c5, vdup_lane_s8(va2, 5));
                vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c5));
                vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c5));
                const int16x8_t vprod3x01234567c5 = vmull_s8(vb01234567c5, vdup_lane_s8(va3, 5));
                vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c5));
                vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c5));

                if (k > 6 * sizeof(int8_t)) {
                  const int8x8_t vb01234567c6 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

                  const int16x8_t vprod0x01234567c6 = vmull_s8(vb01234567c6, vdup_lane_s8(va0, 6));
                  vacc0x0123 = vaddw_s16(vacc0x0123, vget_low_s16(vprod0x01234567c6));
                  vacc0x4567 = vaddw_s16(vacc0x4567, vget_high_s16(vprod0x01234567c6));
                  const int16x8_t vprod1x01234567c6 = vmull_s8(vb01234567c6, vdup_lane_s8(va1, 6));
                  vacc1x0123 = vaddw_s16(vacc1x0123, vget_low_s16(vprod1x01234567c6));
                  vacc1x4567 = vaddw_s16(vacc1x4567, vget_high_s16(vprod1x01234567c6));
                  const int16x8_t vprod2x01234567c6 = vmull_s8(vb01234567c6, vdup_lane_s8(va2, 6));
                  vacc2x0123 = vaddw_s16(vacc2x0123, vget_low_s16(vprod2x01234567c6));
                  vacc2x4567 = vaddw_s16(vacc2x4567, vget_high_s16(vprod2x01234567c6));
                  const int16x8_t vprod3x01234567c6 = vmull_s8(vb01234567c6, vdup_lane_s8(va3, 6));
                  vacc3x0123 = vaddw_s16(vacc3x0123, vget_low_s16(vprod3x01234567c6));
                  vacc3x4567 = vaddw_s16(vacc3x4567, vget_high_s16(vprod3x01234567c6));
                }
              }
            }
          }
        }
      }
    }

    // Post-accumulation work
    const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
    const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);

    vacc0x0123 = vshlq_s32(vacc0x0123, vright_pre_shift);
    vacc0x4567 = vshlq_s32(vacc0x4567, vright_pre_shift);
    vacc1x0123 = vshlq_s32(vacc1x0123, vright_pre_shift);
    vacc1x4567 = vshlq_s32(vacc1x4567, vright_pre_shift);
    vacc2x0123 = vshlq_s32(vacc2x0123, vright_pre_shift);
    vacc2x4567 = vshlq_s32(vacc2x4567, vright_pre_shift);
    vacc3x0123 = vshlq_s32(vacc3x0123, vright_pre_shift);
    vacc3x4567 = vshlq_s32(vacc3x4567, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);
    vacc1x0123 = vqdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqdmulhq_s32(vacc1x4567, vmultiplier);
    vacc2x0123 = vqdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqdmulhq_s32(vacc2x4567, vmultiplier);
    vacc3x0123 = vqdmulhq_s32(vacc3x0123, vmultiplier);
    vacc3x4567 = vqdmulhq_s32(vacc3x4567, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_post_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_post_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_post_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_post_shift);
    vacc3x0123 = vrshlq_s32(vacc3x0123, vright_post_shift);
    vacc3x4567 = vrshlq_s32(vacc3x4567, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
    int8x16_t vout2x01234567_3x01234567 = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc3x01234567);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
    int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc3x01234567));
#endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->rndnu_neon.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->rndnu_neon.output_max);

    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);
    vout2x01234567_3x01234567 = vmaxq_s8(vout2x01234567_3x01234567, voutput_min);

    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);
    vout2x01234567_3x01234567 = vminq_s8(vout2x01234567_3x01234567, voutput_max);

    if (nc >= 8) {
      // Main case where there the 8 columns fit in the destination.
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));
      vst1_s8(c2 + 0, vget_low_s8(vout2x01234567_3x01234567));
      vst1_s8(c3 + 0, vget_high_s8(vout2x01234567_3x01234567));

      // Advance to the next 8 columns.
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      // Final case where not all of the 8 columns fit in the destination.
      if (nc & 4) {
        vst1q_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c1, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c2, 1), vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32(__builtin_assume_aligned(c3, 1), vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c1, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c2, 1), vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16(__builtin_assume_aligned(c3, 1), vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}
