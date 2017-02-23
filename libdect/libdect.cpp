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

#include <stdint.h>
#include <iostream>
#ifndef _MSC_VER
#include "config.h"
#ifdef __GNUC__
#define EXPORT __attribute__ ((visibility ("default")))
#else
#define EXPORT 
#endif
#else
#ifndef HAS_OPENCL
#define HAS_OPENCL 1
#endif
#define EXPORT __declspec(dllexport)
#endif

#if HAS_OPENCL
int opencl_get_device_count();
const char *opencl_get_device_name(int idx);

int dect_algo_opencl(int enhanced,
	const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	uint8_t *x, uint8_t *y, uint8_t *z,
	size_t outsize,
	float min_step,
	int16_t *m,
	float mr,
	int idx_adjust);
#else
int opencl_get_device_count()
{
	return 0;
}

const char *opencl_get_device_name(int idx)
{
	(void)idx;
	return NULL;
}
#endif

int dect_algo_cpu_iter(int enhanced,
	const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	uint8_t *x, uint8_t *y, uint8_t *z,
	size_t outsize,
	float min_step,
	int16_t *m,
	float mr,
	int idx_adjust);

int dect_algo_simul(int enhanced,
	const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	uint8_t *x, uint8_t *y, uint8_t *z,
	size_t out_size,
	float min_step,
	int16_t *m,
	float mr,
	int idx_adjust);

#if HAS_OPENCL
int opencl_init(int platform, int enhanced);
#else
int opencl_init(int platform, int enhanced)
{
	(void)platform;
	(void)enhanced;
	return -1;
}
#endif

EXPORT int dect_getDeviceCount()
{
	return 2 + opencl_get_device_count();
}

EXPORT const char *dect_getDeviceName(int idx)
{
	switch (idx)
	{
	case 0:
		return "CPU";
	case 1:
		return "CPU using simultaneous equations (fast but inaccurate)";
	default:
		return opencl_get_device_name(idx - 2);
	}
}

EXPORT int dect_initDevice(int idx, int enhanced)
{
	if (idx >= 2)
		return opencl_init(idx - 2, enhanced);
	return 0;
}

EXPORT int dect_process(
	int device_id, int enhanced,
	const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	uint8_t *x, uint8_t *y, uint8_t *z,
	size_t outsize,
	float min_step,
	int16_t *m,
	float mr,
	int idx_adjust)
{
	switch (device_id)
	{
	case 0:
		return dect_algo_cpu_iter(enhanced,
			a, b, alphaa, betaa, gammaa,
			alphab, betab, gammab, x, y, z, outsize,
			min_step, m, mr, idx_adjust);
	case 1:
		return dect_algo_simul(enhanced,
			a, b, alphaa, betaa, gammaa,
			alphab, betab, gammab, x, y, z, outsize,
			min_step, m, mr, idx_adjust);
	default:
#if HAS_OPENCL
		auto ret = dect_algo_opencl(enhanced,
			a, b, alphaa, betaa, gammaa,
			alphab, betab, gammab, x, y, z, outsize,
			min_step, m, mr, idx_adjust);
		if (ret != 0)
		{
			std::cerr << "ERROR: OpenCL algorithm failed, switching to CPU" << std::endl;

			return dect_algo_cpu_iter(enhanced,
				a, b, alphaa, betaa, gammaa,
				alphab, betab, gammab, x, y, z, outsize,
				min_step, m, mr, idx_adjust);
		}
		return ret;
#else
		std::cerr << "ERROR: Unknown device ID" << std::endl;
		return -1;
#endif
	}
}

/* Create source images from processed images - for testing accuracy
	of various algorithms */
EXPORT int dect_reconstitute(
	const uint8_t *x, const uint8_t *y, const uint8_t *z,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	int16_t *a, int16_t *b,
	size_t outsize,
	int idx_adjust)
{
	for (size_t idx = 0; idx < outsize; idx++)
	{
		float curx = (float)x[idx] / 255.0f;
		float cury = (float)y[idx] / 255.0f;
		float curz = (float)z[idx] / 255.0f;
		
		float cura = curx * alphaa + cury * betaa + curz * gammaa;
		float curb = curx * alphab + cury * betab + curz * gammab;

		auto out_idx = idx;
		if (idx_adjust)
			out_idx = idx_adjust - idx;

		a[out_idx] = (int16_t)cura;
		b[out_idx] = (int16_t)curb;
	}

	return 0;
}
