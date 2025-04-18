// Auto-generated file. Do not edit!
//   Template: src/qs8-vaddc/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/vaddsub.h>


void xnn_qu8_vaddc_minmax_ukernel__neon_ld64_x32(
    size_t n,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_addsub_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  const uint8x8_t va_zero_point = vld1_dup_u8(&params->neon.a_zero_point);
  const int32x4_t va_multiplier = vld1q_dup_s32(&params->neon.a_multiplier);
  const int32x4_t vright_shift = vld1q_dup_s32(&params->neon.right_shift);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->neon.output_min);
  const uint8x16_t voutput_max = vld1q_dup_u8(&params->neon.output_max);

  const int32_t vxb = (int32_t) *input_b - (int32_t) params->neon.b_zero_point;
  const int32_t vb = params->neon.b_multiplier;
  const int32x4_t vbias = vdupq_n_s32(vxb * vb);

  for (; n >= 32 * sizeof(uint8_t); n -= 32 * sizeof(uint8_t)) {
    const uint8x8_t va01234567 = vld1_u8(input_a); input_a += 8;
    const uint8x8_t va89ABCDEF = vld1_u8(input_a); input_a += 8;
    const uint8x8_t vaGHIJKLMN = vld1_u8(input_a); input_a += 8;
    const uint8x8_t vaOPQRSTUV = vld1_u8(input_a); input_a += 8;

    const int16x8_t vxa01234567 = vreinterpretq_s16_u16(vsubl_u8(va01234567, va_zero_point));
    const int16x8_t vxa89ABCDEF = vreinterpretq_s16_u16(vsubl_u8(va89ABCDEF, va_zero_point));
    const int16x8_t vxaGHIJKLMN = vreinterpretq_s16_u16(vsubl_u8(vaGHIJKLMN, va_zero_point));
    const int16x8_t vxaOPQRSTUV = vreinterpretq_s16_u16(vsubl_u8(vaOPQRSTUV, va_zero_point));

    int32x4_t vacc0123 = vmlaq_s32(vbias, vmovl_s16(vget_low_s16(vxa01234567)), va_multiplier);
    int32x4_t vacc4567 = vmlaq_s32(vbias, vmovl_s16(vget_high_s16(vxa01234567)), va_multiplier);
    int32x4_t vacc89AB = vmlaq_s32(vbias, vmovl_s16(vget_low_s16(vxa89ABCDEF)), va_multiplier);
    int32x4_t vaccCDEF = vmlaq_s32(vbias, vmovl_s16(vget_high_s16(vxa89ABCDEF)), va_multiplier);
    int32x4_t vaccGHIJ = vmlaq_s32(vbias, vmovl_s16(vget_low_s16(vxaGHIJKLMN)), va_multiplier);
    int32x4_t vaccKLMN = vmlaq_s32(vbias, vmovl_s16(vget_high_s16(vxaGHIJKLMN)), va_multiplier);
    int32x4_t vaccOPQR = vmlaq_s32(vbias, vmovl_s16(vget_low_s16(vxaOPQRSTUV)), va_multiplier);
    int32x4_t vaccSTUV = vmlaq_s32(vbias, vmovl_s16(vget_high_s16(vxaOPQRSTUV)), va_multiplier);

    vacc0123 = vrshlq_s32(vacc0123, vright_shift);
    vacc4567 = vrshlq_s32(vacc4567, vright_shift);
    vacc89AB = vrshlq_s32(vacc89AB, vright_shift);
    vaccCDEF = vrshlq_s32(vaccCDEF, vright_shift);
    vaccGHIJ = vrshlq_s32(vaccGHIJ, vright_shift);
    vaccKLMN = vrshlq_s32(vaccKLMN, vright_shift);
    vaccOPQR = vrshlq_s32(vaccOPQR, vright_shift);
    vaccSTUV = vrshlq_s32(vaccSTUV, vright_shift);

    const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);
    const int16x8_t vacc89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF)), voutput_zero_point);
    const int16x8_t vaccGHIJKLMN = vqaddq_s16(vcombine_s16(vqmovn_s32(vaccGHIJ), vqmovn_s32(vaccKLMN)), voutput_zero_point);
    const int16x8_t vaccOPQRSTUV = vqaddq_s16(vcombine_s16(vqmovn_s32(vaccOPQR), vqmovn_s32(vaccSTUV)), voutput_zero_point);

    uint8x16_t vout0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc01234567), vqmovun_s16(vacc89ABCDEF));
    uint8x16_t voutGHIJKLMNOPQRSTUV = vcombine_u8(vqmovun_s16(vaccGHIJKLMN), vqmovun_s16(vaccOPQRSTUV));

    vout0123456789ABCDEF = vmaxq_u8(vout0123456789ABCDEF, voutput_min);
    voutGHIJKLMNOPQRSTUV = vmaxq_u8(voutGHIJKLMNOPQRSTUV, voutput_min);

    vout0123456789ABCDEF = vminq_u8(vout0123456789ABCDEF, voutput_max);
    voutGHIJKLMNOPQRSTUV = vminq_u8(voutGHIJKLMNOPQRSTUV, voutput_max);

    vst1q_u8(output, vout0123456789ABCDEF); output += 16;
    vst1q_u8(output, voutGHIJKLMNOPQRSTUV); output += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const uint8x8_t va01234567 = vld1_u8(input_a); input_a += 8;

      const int16x8_t vxa01234567 = vreinterpretq_s16_u16(vsubl_u8(va01234567, va_zero_point));

      int32x4_t vacc0123 = vmlaq_s32(vbias, vmovl_s16(vget_low_s16(vxa01234567)), va_multiplier);
      int32x4_t vacc4567 = vmlaq_s32(vbias, vmovl_s16(vget_high_s16(vxa01234567)), va_multiplier);

      vacc0123 = vrshlq_s32(vacc0123, vright_shift);
      vacc4567 = vrshlq_s32(vacc4567, vright_shift);

      const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);

      uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
      vout01234567 = vmax_u8(vout01234567, vget_low_u8(voutput_min));
      vout01234567 = vmin_u8(vout01234567, vget_low_u8(voutput_max));

      if XNN_LIKELY(n >= (8 * sizeof(uint8_t))) {
        vst1_u8(output, vout01234567); output += 8;
        n -= 8 * sizeof(uint8_t);
      } else {
        if (n & (4 * sizeof(uint8_t))) {
          vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_u8(vout01234567), 0); output += 4;
          vout01234567 = vext_u8(vout01234567, vout01234567, 4);
        }
        if (n & (2 * sizeof(uint8_t))) {
          vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_u8(vout01234567), 0); output += 2;
          vout01234567 = vext_u8(vout01234567, vout01234567, 2);
        }
        if (n & (1 * sizeof(uint8_t))) {
          vst1_lane_u8(output, vout01234567, 0);
        }
        n = 0;
      }
    } while (n != 0);
  }
}
