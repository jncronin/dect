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

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdint.h>
#include <string.h>
#include <iostream>
#include "config.h"
#ifndef _MSC_VER
#ifdef __GNUC__
#define EXPORT __attribute__ ((visibility ("default")))
#define RESTRICT __restrict
#else
#define EXPORT 
#endif
#else
#define RESTRICT __restrict
#ifndef HAS_OPENCL
#define HAS_OPENCL 1
#endif
#define EXPORT __declspec(dllexport)
#endif

#include "git.version.h"
static const char *prettyversion = "v0.3";

#define IN_LIBDECT
#include "libdect.h"

static int _use_single_fp = 0;
static libdect_output_type _otype = libdect_output_type::u8;

#if HAS_OPENCL
int opencl_get_device_count();
const char *opencl_get_device_name(int idx);

int dect_algo_opencl(int enhanced,
	const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	void *x, void *y, void *z,
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

EXPORT const char *dect_getVersion()
{
	// parse gitversion string to something readable
	const int BUFSIZE = 128;

	const char *ptr = NULL;
	const char *s = gitversion;
	for (; *s; s++)
	{
		if (!isspace(*s))
		{
			ptr = s;
			break;
		}
	}

	char *vstr = (char*)malloc(BUFSIZE);
	char *vptr = vstr;
	if (ptr)
	{
		for (int i = 0; i < 7 && *ptr; i++, ptr++)
			*vptr++ = *ptr;
	}
	*vptr++ = '\0';
	strcat(vstr, " ");
	strcat(vstr, prettyversion);

	return vstr;
}

static int dect_algo_cpu_iter(int enhanced,
	const int16_t * RESTRICT a, const int16_t * RESTRICT b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	void * RESTRICT x,
	void * RESTRICT y,
	void * RESTRICT z,
	size_t outsize,
	float min_step,
	int16_t * RESTRICT m,
	float mr,
	int idx_adjust);

/* Use to allow multiple cpu prototypes to be defined */
#define CPU_PROTOTYPE(osig, otype)	int dect_algo_cpuf##osig##_iter(int enhanced, \
	const int16_t * RESTRICT a, const int16_t * RESTRICT b, \
	float alphaa, float betaa, float gammaa, \
	float alphab, float betab, float gammab, \
	otype * RESTRICT x, \
	otype * RESTRICT y, \
	otype * RESTRICT z, \
	size_t outsize, \
	float min_step, \
	int16_t * RESTRICT m, \
	float mr, \
	int idx_adjust); \
	\
	int dect_algo_cpud##osig##_iter(int enhanced, \
	const int16_t * RESTRICT a, const int16_t * RESTRICT b, \
	float alphaa, float betaa, float gammaa, \
	float alphab, float betab, float gammab, \
	otype * RESTRICT x, \
	otype * RESTRICT y, \
	otype * RESTRICT z, \
	size_t outsize, \
	float min_step, \
	int16_t * RESTRICT m, \
	float mr, \
	int idx_adjust);

CPU_PROTOTYPE(8, uint8_t)
CPU_PROTOTYPE(16, uint16_t)
CPU_PROTOTYPE(f32, float)
CPU_PROTOTYPE(f64, double)

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
int opencl_init(int platform, int enhanced,
	int use_single_fp, libdect_output_type otype);
#else
int opencl_init(int platform, int enhanced,
	int use_single_fp, libdect_output_type otype)
{
	(void)platform;
	(void)enhanced;
	(void)use_single_fp;
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

EXPORT int dect_initDevice(int idx, int enhanced,
	int use_single_fp, libdect_output_type otype)
{
	if (idx >= 2)
		opencl_init(idx - 2, enhanced, use_single_fp,
			otype);
	_use_single_fp = use_single_fp;
	_otype = otype;
	return 0;
}

static int dect_algo_cpu_iter(int enhanced,
	const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	void *x, void *y, void *z,
	size_t pix_count,
	float min_step,
	int16_t *m,
	float mr,
	int idx_adjust)
{
	if (_use_single_fp)
	{
		switch(_otype)
		{
			case libdect_output_type::u16:
				return dect_algo_cpuf16_iter(enhanced,
					a, b, alphaa, betaa, gammaa,
					alphab, betab, gammab,
					(uint16_t*)x, (uint16_t*)y, (uint16_t*)z,
					pix_count,
					min_step, m, mr, idx_adjust);
			case libdect_output_type::u8:
				return dect_algo_cpuf8_iter(enhanced,
					a, b, alphaa, betaa, gammaa,
					alphab, betab, gammab,
					(uint8_t*)x, (uint8_t*)y, (uint8_t*)z,
					pix_count,
					min_step, m, mr, idx_adjust);
			case libdect_output_type::f32:
				return dect_algo_cpuff32_iter(enhanced,
					a, b, alphaa, betaa, gammaa,
					alphab, betab, gammab,
					(float*)x, (float*)y, (float*)z,
					pix_count,
					min_step, m, mr, idx_adjust);
			case libdect_output_type::f64:
				return dect_algo_cpuff64_iter(enhanced,
					a, b, alphaa, betaa, gammaa,
					alphab, betab, gammab,
					(double*)x, (double*)y, (double*)z,
					pix_count,
					min_step, m, mr, idx_adjust);
		}
	}
	else
	{
		switch (_otype)
		{
			case libdect_output_type::u16:
				return dect_algo_cpud16_iter(enhanced,
					a, b, alphaa, betaa, gammaa,
					alphab, betab, gammab,
					(uint16_t*)x, (uint16_t*)y, (uint16_t*)z,
					pix_count,
					min_step, m, mr, idx_adjust);
			case libdect_output_type::u8:
				return dect_algo_cpud8_iter(enhanced,
					a, b, alphaa, betaa, gammaa,
					alphab, betab, gammab,
					(uint8_t*)x, (uint8_t*)y, (uint8_t*)z,
					pix_count,
					min_step, m, mr, idx_adjust);
			case libdect_output_type::f32:
				return dect_algo_cpudf32_iter(enhanced,
					a, b, alphaa, betaa, gammaa,
					alphab, betab, gammab,
					(float*)x, (float*)y, (float*)z,
					pix_count,
					min_step, m, mr, idx_adjust);
			case libdect_output_type::f64:
				return dect_algo_cpudf64_iter(enhanced,
					a, b, alphaa, betaa, gammaa,
					alphab, betab, gammab,
					(double*)x, (double*)y, (double*)z,
					pix_count,
					min_step, m, mr, idx_adjust);
		}
	}

	/* Shouldn't get here */
	return -1;
}

EXPORT int dect_process(
	int device_id, int enhanced,
	const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	void *x, void *y, void *z,
	size_t pix_count,
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
			alphab, betab, gammab,
			x, y, z,
			pix_count,
			min_step, m, mr, idx_adjust);
		
	case 1:
		return dect_algo_simul(enhanced,
			a, b, alphaa, betaa, gammaa,
			alphab, betab, gammab, (uint8_t*)x, (uint8_t*)y, (uint8_t*)z, pix_count,
			min_step, m, mr, idx_adjust);
	default:
#if HAS_OPENCL
		auto ret = dect_algo_opencl(enhanced,
			a, b, alphaa, betaa, gammaa,
			alphab, betab, gammab, x, y, z, pix_count,
			min_step, m, mr, idx_adjust);
		if (ret != 0)
		{
			std::cerr << "ERROR: OpenCL algorithm failed, switching to CPU" << std::endl;

			return dect_algo_cpu_iter(enhanced,
				a, b, alphaa, betaa, gammaa,
				alphab, betab, gammab, x, y, z, pix_count,
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
