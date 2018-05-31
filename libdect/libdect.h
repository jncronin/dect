/* Copyright (C) 2016 by John Cronin
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

#pragma once

#ifndef LIBDECT_H
#define LIBDECT_H

#include <stdint.h>

enum libdect_output_type
{
	u8, u16, f32, f64
};

#ifndef IN_LIBDECT
int dect_getDeviceCount();
const char *dect_getVersion();
const char *dect_getDeviceName(int idx);
int dect_initDevice(int idx, int enhanced, int use_single_fp,
	libdect_output_type otype);

int dect_process(
	int device_id,
	int enhanced,
	const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	void *x, void *y, void *z,
	size_t pix_count,
	float min_step,
	int16_t *m,
	float mr,
	int idx_adjust);

int dect_reconstitute(
	const uint8_t *x, const uint8_t *y, const uint8_t *z,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	int16_t *a, int16_t *b,
	size_t outsize,
	int idx_adjust);

#endif

#endif
