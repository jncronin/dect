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

kernel void dect(global short *a, global short *b,
	const double alphaa, const double betaa, const double gammaa,
	const double alphab, const double betab, const double gammab,
	global uchar *x, global uchar *y, global uchar *z,
	const double min_step,
	global short *m,
	const double mr,
	const int do_merge,
	const int idx_adjust)
{
	size_t idx = get_global_id(0);

	double dA = (double)a[idx];
	double dB = (double)b[idx];

	/* First, iterate through ratio and ab at 0.1 intervals
	to ensure we don't miss an approximate solution, then
	iterate to find the actual best value - this prevents us
	finding islands of solutions which aren't necessarily the best
	solutions */

	double best_err = 1000000.0;
	double best_ab = 0.66;
	double best_ratio = 0.5;

	for (double test_ab = 0.0; test_ab <= 1.0; test_ab += 0.1)
	{
		for (double test_ratio = 0.0; test_ratio <= 1.0; test_ratio += 0.1)
		{
			double cur_a = test_ab * test_ratio;
			double cur_b = test_ab * (1.0 - test_ratio);
			double cur_c = 1.0 - test_ab;

			double dA_est = alphaa * cur_a + betaa * cur_b + gammaa * cur_c;
			double dB_est = alphab * cur_a + betab * cur_b + gammab * cur_c;

			//double dA_err = (double)pow(dA_est - dA, 2.0);
			//double dB_err = (double)pow(dB_est - dB, 2.0);
			double dA_err = (dA_est - dA) * (dA_est - dA);
			double dB_err = (dB_est - dB) * (dB_est - dB);

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
	double cur_error = 5000.0 * 5000.0;
	double cur_step = 0.05;
	double cur_ratio = best_ratio;
	double cur_ab = best_ab;

	while (cur_step >= min_step)
	{
		double min_err;
		double min_ab;
		double min_ratio;

		for (int i = 0; i < 4; i++)
		{
			double new_ab, new_ratio;
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

			double cur_a = new_ab * new_ratio;
			double cur_b = new_ab * (1.0 - new_ratio);
			double cur_c = 1.0 - new_ab;

			double dA_est = alphaa * cur_a + betaa * cur_b + gammaa * cur_c;
			double dB_est = alphab * cur_a + betab * cur_b + gammab * cur_c;

			//double dA_err = (double)pow(dA_est - dA, 2.0);
			//double dB_err = (double)pow(dB_est - dB, 2.0);
			double dA_err = (dA_est - dA) * (dA_est - dA);
			double dB_err = (dB_est - dB) * (dB_est - dB);

			double tot_err = dA_err + dB_err;

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

	uchar best_a = (uchar)floor(cur_ab * cur_ratio * 255.0);
	uchar best_b = (uchar)floor(cur_ab * (1.0 - cur_ratio) * 255.0);
	uchar best_c = (uchar)floor((1.0 - cur_ab) * 255.0);

	x[idx] = best_a;
	y[idx] = best_b;
	z[idx] = best_c;

	if(do_merge)
		m[idx] = (short)(dA * mr + dB * (1.0 - mr));
}

kernel void dect2(global short *a, global short *b,
	double alphaa, double betaa, double gammaa,
	double alphab, double betab, double gammab,
	global uchar *x, global uchar *y, global uchar *z,
	double min_step,
	global short *m,
	double mr,
	int do_merge,
	int idx_adjust)
{
	size_t idx = get_global_id(0);

	double dA = a[idx];
	double dB = b[idx];

	double tot_best_a = 0.0;
	double tot_best_b = 0.0;
	double tot_best_c = 0.0;

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
		double calphaa, cbetaa, cgammaa;
		double calphab, cbetab, cgammab;

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

				double dA_err = (dA_est - dA) * (dA_est - dA);
				double dB_err = (dB_est - dB) * (dB_est - dB);

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
		double cur_step = 0.05;

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

		double cur_best_a, cur_best_b, cur_best_c;

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

	uchar best_a = (uchar)floor(tot_best_a / 3.0 * 255.0);
	uchar best_b = (uchar)floor(tot_best_b / 3.0 * 255.0);
	uchar best_c = (uchar)floor(tot_best_c / 3.0 * 255.0);

	x[idx] = best_a;
	y[idx] = best_b;
	z[idx] = best_c;

	if(do_merge)
		m[idx] = (short)(dA * mr + dB * (1.0 - mr));
}

)OPENCL";
