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
#include <math.h>
#include <stddef.h>
#include <algorithm>

#ifndef _MSC_VER
#ifdef __GNUC__
#define RESTRICT __restrict
#endif
#else
#define RESTRICT __restrict
#endif

#ifndef FLOOR_FUNC
#define FLOOR_FUNC floor
#endif

/* Algorithm written with a view to parallelizing with OpenCL
a, b			- input images
alphaa, alphab	- CT density of material 1 in image a and b
betaa, betab	- CT density of material 2 in image a and b
gammaa, gammab	- CT density of material 3 in image a and b
idx				- voxel number
x, y, z			- output images
min_step		- threshold below which to stop algorithm

cur_ratio = cur_a / (cur_a + cur_b)
cur_ab = cur_a + cur_b

Therefore:
cur_a = cur_ab * cur_ratio
cur_b = cur_ab * (1 - cur_ratio)
cur_c = 1 - cur_ab

Thus, as long as we clamp cur_ab and cur_ratio to [0,1],
cur_a, cur_b and cur_c will all be in the range [0,1] and
additionally sum to 1.

We choose a position (cur_ratio, cur_ab) in a 2D plane
and progressively move this point in either +x, +y, -x or -y
directions by the value cur_step (clamped to [0,1])

If the error sum of squares in voxel densities in A and B
is reduced by any of these new points, we repeat with the
new point as the base.

If not, we reduce the value of cur_step and repeat with the
current point.

When cur_step < min_step we stop.
*/
static inline
#if _MSC_VER
__forceinline 
#endif
void dect_algo_cpu(int enhanced,
	const int16_t * RESTRICT a, const int16_t * RESTRICT b,
	FPTYPE alphaa, FPTYPE betaa, FPTYPE gammaa,
	FPTYPE alphab, FPTYPE betab, FPTYPE gammab,
	int idx,
	OTYPE * RESTRICT x,
	OTYPE * RESTRICT y,
	OTYPE * RESTRICT z,
	FPTYPE min_step,
	int16_t * RESTRICT m,
	FPTYPE mr,
	int idx_adjust)
{
#ifdef __GNUC__
#ifdef __x86_64__
	__builtin_assume_aligned(a, 16);
	__builtin_assume_aligned(b, 16);
	__builtin_assume_aligned(x, 16);
	__builtin_assume_aligned(y, 16);
	__builtin_assume_aligned(z, 16);
	__builtin_assume_aligned(m, 16);
#endif
#endif
	FPTYPE dA = a[idx];
	FPTYPE dB = b[idx];

	/* Clamp actual value to the max/min of the input values */
	FPTYPE maxA = std::max(alphaa, std::max(betaa, gammaa));
	FPTYPE minA = std::min(alphaa, std::min(betaa, gammaa));
	FPTYPE maxB = std::max(alphab, std::max(betab, gammab));
	FPTYPE minB = std::min(alphab, std::min(betab, gammab));
	dA = std::clamp(dA, minA, maxA);
	dB = std::clamp(dB, minB, maxB);

	FPTYPE tot_best_a = 0.0;
	FPTYPE tot_best_b = 0.0;
	FPTYPE tot_best_c = 0.0;

	/* in the case of an enhanced algorithm, we do the same
	as the standard but permutate a, b, and c through
	the orders:
	a, b, c
	c, a, b
	b, c, a
	so that each spends two iterations as part of
	cur_ab and one as not (i.e. 1 - cur_ratio)
	This avoids bias towards/against a particular
	material
	We average out the values at the end */

	for (int i = 0; i < enhanced; i++)
	{
		FPTYPE cur_ratio = 0.5;
		FPTYPE cur_ab = 0.66;
		FPTYPE cur_step = 0.25;
		FPTYPE cur_error = 5000.0 * 5000.0;

		FPTYPE calphaa, cbetaa, cgammaa;
		FPTYPE calphab, cbetab, cgammab;

		switch (i)
		{
		case 0:
			calphaa = alphaa;
			cbetaa = betaa;
			cgammaa = gammaa;
			calphab = alphab;
			cbetab = betab;
			cgammab = gammab;
			break;
		case 1:
			calphaa = gammaa;
			cbetaa = alphaa;
			cgammaa = betaa;
			calphab = gammab;
			cbetab = alphab;
			cgammab = betab;
			break;
		case 2:
			calphaa = betaa;
			cbetaa = gammaa;
			cgammaa = alphaa;
			calphab = betab;
			cbetab = gammab;
			cgammab = alphab;
			break;
		}

		/* First, iterate through ratio and ab at 0.1 intervals
		to ensure we don't miss an approximate solution, then
		iterate to find the actual best value - this prevents us
		finding islands of solutions which aren't necessarily the best
		solutions */

		FPTYPE best_err = 5000.0 * 5000.0;
		FPTYPE best_ab = 0.0;
		FPTYPE best_ratio = 0.0;

		for (FPTYPE test_ab = 0.0; test_ab <= 1.0; test_ab += 0.1)
		{
			for (FPTYPE test_ratio = 0.0; test_ratio <= 1.0; test_ratio += 0.1)
			{
				FPTYPE cur_a = test_ab * test_ratio;
				FPTYPE cur_b = test_ab * (1.0 - test_ratio);
				FPTYPE cur_c = 1.0 - cur_a - cur_b;

				FPTYPE dA_est = calphaa * cur_a + cbetaa * cur_b + cgammaa * cur_c;
				FPTYPE dB_est = calphab * cur_a + cbetab * cur_b + cgammab * cur_c;

				FPTYPE dA_err = (FPTYPE)pow(dA_est - dA, 2);
				FPTYPE dB_err = (FPTYPE)pow(dB_est - dB, 2);

				FPTYPE tot_err = dA_err + dB_err;

				if (tot_err < best_err)
				{
					best_err = tot_err;
					best_ab = test_ab;
					best_ratio = test_ratio;
				}
			}
		}

		/* Now do an iterative search to find the best values */
		cur_step = 0.05;
		cur_ratio = best_ratio;
		cur_ab = best_ab;

		while (cur_step >= min_step)
		{
			FPTYPE min_err;
			FPTYPE min_ab;
			FPTYPE min_ratio;

			for (int j = 0; j < 4; j++)
			{
				FPTYPE new_ab, new_ratio;
				switch (j)
				{
				case 0:
					new_ab = cur_ab + cur_step;
					new_ratio = cur_ratio;
					break;
				case 1:
					new_ab = cur_ab;
					new_ratio = cur_ratio + cur_step;
					break;
				case 2:
					new_ab = cur_ab - cur_step;
					new_ratio = cur_ratio;
					break;
				case 3:
					new_ab = cur_ab;
					new_ratio = cur_ratio - cur_step;
					break;
				}

				if (new_ab < 0.0)
					new_ab = 0.0;
				if (new_ab > 1.0)
					new_ab = 1.0;
				if (new_ratio < 0.0)
					new_ratio = 0.0;
				if (new_ratio > 1.0)
					new_ratio = 1.0;

				FPTYPE cur_a = new_ab * new_ratio;
				FPTYPE cur_b = new_ab * (1.0 - new_ratio);
				FPTYPE cur_c = 1.0 - new_ab;

				FPTYPE dA_est = calphaa * cur_a + cbetaa * cur_b + cgammaa * cur_c;
				FPTYPE dB_est = calphab * cur_a + cbetab * cur_b + cgammab * cur_c;

				//FPTYPE dA_err = (FPTYPE)pow(dA_est - dA, 2.0);
				//FPTYPE dB_err = (FPTYPE)pow(dB_est - dB, 2.0);
				FPTYPE dA_err = (dA_est - dA) * (dA_est - dA);
				FPTYPE dB_err = (dB_est - dB) * (dB_est - dB);

				FPTYPE tot_err = dA_err + dB_err;

				if (j == 0 || tot_err < min_err)
				{
					min_err = tot_err;
					min_ratio = new_ratio;
					min_ab = new_ab;
				}
			}

			if (min_err < cur_error)
			{
				cur_ratio = min_ratio;
				cur_ab = min_ab;
				cur_error = min_err;
			}
			else
			{
				cur_step = cur_step / 2.0;
			}
		}

		FPTYPE cur_best_a, cur_best_b, cur_best_c;

		switch (i)
		{
		case 0:
			cur_best_a = cur_ab * cur_ratio;
			cur_best_b = cur_ab * (1.0 - cur_ratio);
			cur_best_c = 1.0 - cur_ab;
			break;
		case 1:
			cur_best_c = cur_ab * cur_ratio;
			cur_best_a = cur_ab * (1.0 - cur_ratio);
			cur_best_b = 1.0 - cur_ab;
			break;
		case 2:
			cur_best_b = cur_ab * cur_ratio;
			cur_best_c = cur_ab * (1.0 - cur_ratio);
			cur_best_a = 1.0 - cur_ab;
			break;
		}

		tot_best_a += cur_best_a;
		tot_best_b += cur_best_b;
		tot_best_c += cur_best_c;
	}

	if (enhanced > 1)
	{
		tot_best_a /= enhanced;
		tot_best_b /= enhanced;
		tot_best_c /= enhanced;
	}

	if (idx_adjust)
		idx = idx_adjust - idx;

	OTYPE best_a = (OTYPE)FLOOR_FUNC(tot_best_a * OTYPE_MAX);
	OTYPE best_b = (OTYPE)FLOOR_FUNC(tot_best_b * OTYPE_MAX);
	OTYPE best_c = (OTYPE)FLOOR_FUNC(tot_best_c * OTYPE_MAX);

	x[idx] = best_a;
	y[idx] = best_b;
	z[idx] = best_c;

	if (m)
	{
		m[idx] = (int16_t)((FPTYPE)a[idx] * mr + (FPTYPE)b[idx] * (1.0 - mr));
	}
}

int dect_algo_cpu_iter(int enhanced,
	const int16_t * RESTRICT a, const int16_t * RESTRICT b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	OTYPE * RESTRICT x,
	OTYPE * RESTRICT y,
	OTYPE * RESTRICT z,
	size_t pix_count,
	float min_step,
	int16_t * RESTRICT m,
	float mr,
	int idx_adjust)
{
#if _MSC_VER >= 1700
#pragma loop (hint_parallel(0))
#endif
	for (auto i = 0; i < (int)pix_count; i++)
		dect_algo_cpu(enhanced, a, b, alphaa, betaa, gammaa,
			alphab, betab, gammab, i, x, y, z, min_step,
			m, mr, idx_adjust);
	return 0;
}
