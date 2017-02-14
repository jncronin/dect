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
static void dect_algo_cpu(int enhanced,
	const int16_t *a, const int16_t *b,
	double alphaa, double betaa, double gammaa,
	double alphab, double betab, double gammab,
	int idx,
	uint8_t *x, uint8_t *y, uint8_t *z,
	double min_step,
	int16_t *m,
	double mr,
	int idx_adjust)
{
	double dA = a[idx];
	double dB = b[idx];

	double tot_best_a = 0.0;
	double tot_best_b = 0.0;
	double tot_best_c = 0.0;

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
		double cur_ratio = 0.5;
		double cur_ab = 0.66;
		double cur_step = 0.25;
		double cur_error = 5000.0 * 5000.0;

		double calphaa, cbetaa, cgammaa;
		double calphab, cbetab, cgammab;

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

		double best_err = 5000.0 * 5000.0;
		double best_ab = 0.0;
		double best_ratio = 0.0;

		for (double test_ab = 0.0; test_ab <= 1.0; test_ab += 0.1)
		{
			for (double test_ratio = 0.0; test_ratio <= 1.0; test_ratio += 0.1)
			{
				double cur_a = test_ab * test_ratio;
				double cur_b = test_ab * (1.0 - test_ratio);
				double cur_c = 1.0 - cur_a - cur_b;

				double dA_est = calphaa * cur_a + cbetaa * cur_b + cgammaa * cur_c;
				double dB_est = calphab * cur_a + cbetab * cur_b + cgammab * cur_c;

				double dA_err = (double)pow(dA_est - dA, 2);
				double dB_err = (double)pow(dB_est - dB, 2);

				double tot_err = dA_err + dB_err;

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
			double min_err;
			double min_ab;
			double min_ratio;

			for (int j = 0; j < 4; j++)
			{
				double new_ab, new_ratio;
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

				double cur_a = new_ab * new_ratio;
				double cur_b = new_ab * (1.0 - new_ratio);
				double cur_c = 1.0 - new_ab;

				double dA_est = calphaa * cur_a + cbetaa * cur_b + cgammaa * cur_c;
				double dB_est = calphab * cur_a + cbetab * cur_b + cgammab * cur_c;

				//double dA_err = (double)pow(dA_est - dA, 2.0);
				//double dB_err = (double)pow(dB_est - dB, 2.0);
				double dA_err = (dA_est - dA) * (dA_est - dA);
				double dB_err = (dB_est - dB) * (dB_est - dB);

				double tot_err = dA_err + dB_err;

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

		double cur_best_a, cur_best_b, cur_best_c;

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

	uint8_t best_a = (uint8_t)floor(tot_best_a * 255.0);
	uint8_t best_b = (uint8_t)floor(tot_best_b * 255.0);
	uint8_t best_c = (uint8_t)floor(tot_best_c * 255.0);

	x[idx] = best_a;
	y[idx] = best_b;
	z[idx] = best_c;

	if (m)
		m[idx] = (uint16_t)(dA * mr + dB * (1.0f - mr));
}

int dect_algo_cpu_iter(int enhanced,
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
	for (auto i = 0; i < (int)outsize; i++)
		dect_algo_cpu(enhanced, a, b, alphaa, betaa, gammaa,
			alphab, betab, gammab, i, x, y, z, min_step,
			m, mr, idx_adjust);
	return 0;
}
