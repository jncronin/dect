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

const std::string ks = R"OPENCL(

#ifndef FLOOR_FUNC
#define FLOOR_FUNC floor
#endif

kernel void dect(global short *a, global short *b,
	const FPTYPE alphaa, const FPTYPE betaa, const FPTYPE gammaa,
	const FPTYPE alphab, const FPTYPE betab, const FPTYPE gammab,
	global OTYPE *x, global OTYPE *y, global OTYPE *z,
	const FPTYPE min_step,
	global short *m,
	const FPTYPE mr,
	const int do_merge,
	const int idx_adjust)
{
	size_t idx = get_global_id(0);

	FPTYPE dA = (FPTYPE)a[idx];
	FPTYPE dB = (FPTYPE)b[idx];

	/* Clamp actual value to the max/min of the input values */
	FPTYPE maxA = max(alphaa, max(betaa, gammaa));
	FPTYPE minA = min(alphaa, min(betaa, gammaa));
	FPTYPE maxB = max(alphab, max(betab, gammab));
	FPTYPE minB = min(alphab, min(betab, gammab));
	dA = clamp(dA, minA, maxA);
	dB = clamp(dB, minB, maxB);

	/* First, iterate through ratio and ab at 0.1 intervals
	to ensure we don't miss an approximate solution, then
	iterate to find the actual best value - this prevents us
	finding islands of solutions which aren't necessarily the best
	solutions */

	FPTYPE best_err = 1000000.0;
	FPTYPE best_ab = 0.66;
	FPTYPE best_ratio = 0.5;

	for (FPTYPE test_ab = 0.0; test_ab <= 1.0; test_ab += 0.1)
	{
		for (FPTYPE test_ratio = 0.0; test_ratio <= 1.0; test_ratio += 0.1)
		{
			FPTYPE cur_a = test_ab * test_ratio;
			FPTYPE cur_b = test_ab * (1.0 - test_ratio);
			FPTYPE cur_c = 1.0 - test_ab;

			FPTYPE dA_est = alphaa * cur_a + betaa * cur_b + gammaa * cur_c;
			FPTYPE dB_est = alphab * cur_a + betab * cur_b + gammab * cur_c;

			//FPTYPE dA_err = (FPTYPE)pow(dA_est - dA, 2.0);
			//FPTYPE dB_err = (FPTYPE)pow(dB_est - dB, 2.0);
			FPTYPE dA_err = (dA_est - dA) * (dA_est - dA);
			FPTYPE dB_err = (dB_est - dB) * (dB_est - dB);

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
	FPTYPE cur_error = 5000.0 * 5000.0;
	FPTYPE cur_step = 0.05;
	FPTYPE cur_ratio = best_ratio;
	FPTYPE cur_ab = best_ab;

	while (cur_step >= min_step)
	{
		FPTYPE min_err;
		FPTYPE min_ab;
		FPTYPE min_ratio;

		for (int i = 0; i < 4; i++)
		{
			FPTYPE new_ab, new_ratio;
			switch(i)
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

			if(new_ab < 0.0)
				new_ab = 0.0;
			if(new_ab > 1.0)
				new_ab = 1.0;
			if(new_ratio < 0.0)
				new_ratio = 0.0;
			if(new_ratio > 1.0)
				new_ratio = 1.0;

			FPTYPE cur_a = new_ab * new_ratio;
			FPTYPE cur_b = new_ab * (1.0 - new_ratio);
			FPTYPE cur_c = 1.0 - new_ab;

			FPTYPE dA_est = alphaa * cur_a + betaa * cur_b + gammaa * cur_c;
			FPTYPE dB_est = alphab * cur_a + betab * cur_b + gammab * cur_c;

			//FPTYPE dA_err = (FPTYPE)pow(dA_est - dA, 2.0);
			//FPTYPE dB_err = (FPTYPE)pow(dB_est - dB, 2.0);
			FPTYPE dA_err = (dA_est - dA) * (dA_est - dA);
			FPTYPE dB_err = (dB_est - dB) * (dB_est - dB);

			FPTYPE tot_err = dA_err + dB_err;

			if (i == 0 || tot_err < min_err)
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

	if(idx_adjust)
		idx = idx_adjust - idx;

	OTYPE best_a = (OTYPE)FLOOR_FUNC(cur_ab * cur_ratio * OTYPE_MAX);
	OTYPE best_b = (OTYPE)FLOOR_FUNC(cur_ab * (1.0 - cur_ratio) * OTYPE_MAX);
	OTYPE best_c = (OTYPE)FLOOR_FUNC((1.0 - cur_ab) * OTYPE_MAX);

	x[idx] = best_a;
	y[idx] = best_b;
	z[idx] = best_c;

	if(do_merge)
		m[idx] = (short)(dA * mr + dB * (1.0 - mr));
}

kernel void dect2(global short *a, global short *b,
	FPTYPE alphaa, FPTYPE betaa, FPTYPE gammaa,
	FPTYPE alphab, FPTYPE betab, FPTYPE gammab,
	global OTYPE *x, global OTYPE *y, global OTYPE *z,
	FPTYPE min_step,
	global short *m,
	FPTYPE mr,
	int do_merge,
	int idx_adjust)
{
	size_t idx = get_global_id(0);

	FPTYPE dA = a[idx];
	FPTYPE dB = b[idx];

	/* Clamp actual value to the max/min of the input values */
	FPTYPE maxA = max(alphaa, max(betaa, gammaa));
	FPTYPE minA = min(alphaa, min(betaa, gammaa));
	FPTYPE maxB = max(alphab, max(betab, gammab));
	FPTYPE minB = min(alphab, min(betab, gammab));
	dA = clamp(dA, minA, maxA);
	dB = clamp(dB, minB, maxB);

	FPTYPE tot_best_a = 0.0;
	FPTYPE tot_best_b = 0.0;
	FPTYPE tot_best_c = 0.0;

	/* this is dect() as above, but it permutates the
	a, b and c values through the orders:
		a, b, c
		c, a, b
		b, c, a
	so that each gets an even turn in cur_ratio and cur_ab
	so we don't bias towards one or the other.
	Then average out at the end */
	for(int i = 0; i < 3; i++)
	{
		FPTYPE calphaa, cbetaa, cgammaa;
		FPTYPE calphab, cbetab, cgammab;

		switch(i)
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

				FPTYPE dA_err = (dA_est - dA) * (dA_est - dA);
				FPTYPE dB_err = (dB_est - dB) * (dB_est - dB);

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
		FPTYPE cur_step = 0.05;

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
					new_ab = best_ab + cur_step;
					new_ratio = best_ratio;
					break;
				case 1:
					new_ab = best_ab;
					new_ratio = best_ratio + cur_step;
					break;
				case 2:
					new_ab = best_ab - cur_step;
					new_ratio = best_ratio;
					break;
				case 3:
					new_ab = best_ab;
					new_ratio = best_ratio - cur_step;
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

			if (min_err < best_err)
			{
				best_ratio = min_ratio;
				best_ab = min_ab;
				best_err = min_err;
			}
			else
			{
				cur_step = cur_step / 2.0;
			}
		}

		FPTYPE cur_best_a, cur_best_b, cur_best_c;

		switch(i)
		{
			case 0:
				cur_best_a = best_ab * best_ratio;
				cur_best_b = best_ab * (1.0 - best_ratio);
				cur_best_c = 1.0 - best_ab;
				break;
			case 1:
				cur_best_c = best_ab * best_ratio;
				cur_best_a = best_ab * (1.0 - best_ratio);
				cur_best_b = 1.0 - best_ab;
				break;
			case 2:
				cur_best_b = best_ab * best_ratio;
				cur_best_c = best_ab * (1.0 - best_ratio);
				cur_best_a = 1.0 - best_ab;
				break;
		}

		tot_best_a += cur_best_a;
		tot_best_b += cur_best_b;
		tot_best_c += cur_best_c;
	}

	if(idx_adjust)
		idx = idx_adjust - idx;

	OTYPE best_a = (OTYPE)FLOOR_FUNC(tot_best_a / 3.0 * OTYPE_MAX);
	OTYPE best_b = (OTYPE)FLOOR_FUNC(tot_best_b / 3.0 * OTYPE_MAX);
	OTYPE best_c = (OTYPE)FLOOR_FUNC(tot_best_c / 3.0 * OTYPE_MAX);

	x[idx] = best_a;
	y[idx] = best_b;
	z[idx] = best_c;

	if(do_merge)
		m[idx] = (short)(dA * mr + dB * (1.0 - mr));
}

)OPENCL";
