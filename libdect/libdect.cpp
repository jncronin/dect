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

int opencl_init(int platform, int enhanced);

int dect_getDeviceCount()
{
	return 2 + opencl_get_device_count();
}

const char *dect_getDeviceName(int idx)
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

int dect_initDevice(int idx, int enhanced)
{
	if (idx >= 2)
		return opencl_init(idx - 2, enhanced);
	return 0;
}

int dect_process(
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
	}
}