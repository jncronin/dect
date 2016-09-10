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

kernel void dect(global short *a, global short *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	global uchar *x, global uchar *y, global uchar *z,
	float min_step,
	global short *m,
	float mr,
	int do_merge,
	int idx_adjust)
{
	size_t idx = get_global_id(0);

	float dA = a[idx];
	float dB = b[idx];

	float cur_ratio = 0.5f;
	float cur_ab = 0.66f;
	float cur_step = 0.25f;
	float cur_error = 5000.0f * 5000.0f;

	while (cur_step >= min_step)
	{
		float new_ratio[4];
		float new_ab[4];

		new_ratio[0] = cur_ratio;
		new_ratio[1] = cur_ratio + cur_step;
		new_ratio[2] = cur_ratio;
		new_ratio[3] = cur_ratio - cur_step;

		new_ab[0] = cur_ab + cur_step;
		new_ab[1] = cur_ab;
		new_ab[2] = cur_ab - cur_step;
		new_ab[3] = cur_ab;

		for (int i = 0; i < 4; i++)
		{
			if (new_ratio[i] < 0.0f)
				new_ratio[i] = 0.0f;
			else if (new_ratio[i] > 1.0f)
				new_ratio[i] = 1.0f;

			if (new_ab[i] < 0.0f)
				new_ab[i] = 0.0f;
			else if (new_ab[i] > 1.0f)
				new_ab[i] = 1.0f;
		}

		float new_a[4];
		float new_b[4];

		int min_idx;
		float min_err;

		for (int i = 0; i < 4; i++)
		{
			new_a[i] = new_ab[i] * new_ratio[i];
			new_b[i] = new_ab[i] * (1.0f - new_ratio[i]);
			
			float cur_a = new_a[i];
			float cur_b = new_b[i];
			float cur_c = 1.0f - cur_a - cur_b;

			float dA_est = alphaa * cur_a + betaa * cur_b + gammaa * cur_c;
			float dB_est = alphab * cur_a + betab * cur_b + gammab * cur_c;

			float dA_err = pow(dA_est - dA, 2.0f);
			float dB_err = pow(dB_est - dB, 2.0f);

			float tot_err = dA_err + dB_err;

			if (i == 0 || tot_err < min_err)
			{
				min_idx = i;
				min_err = tot_err;
			}
		}

		if (min_err < cur_error)
		{
			cur_ratio = new_ratio[min_idx];
			cur_ab = new_ab[min_idx];
			cur_error = min_err;
		}
		else
		{
			cur_step = cur_step / 2.0f;
		}
	}

	if(idx_adjust)
		idx = idx_adjust - idx;

	uchar best_a = (uchar)floor(cur_ab * cur_ratio * 255.0f);
	uchar best_b = (uchar)floor(cur_ab * (1.0f - cur_ratio) * 255.0f);
	uchar best_c = (uchar)floor((1.0f - cur_ab) * 255.0f);

	x[idx] = best_a;
	y[idx] = best_b;
	z[idx] = best_c;

	if(do_merge)
		m[idx] = (short)(dA * mr + dB * (1.0f - mr));
}

kernel void dect2(global short *a, global short *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	global uchar *x, global uchar *y, global uchar *z,
	float min_step,
	global short *m,
	float mr,
	int do_merge,
	int idx_adjust)
{
	size_t idx = get_global_id(0);

	float dA = a[idx];
	float dB = b[idx];

	float tot_best_a = 0.0f;
	float tot_best_b = 0.0f;
	float tot_best_c = 0.0f;

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
		float cur_ratio = 0.5f;
		float cur_ab = 0.66f;
		float cur_step = 0.25f;
		float cur_error = 5000.0f * 5000.0f;

		float calphaa, cbetaa, cgammaa;
		float calphab, cbetab, cgammab;

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

		while (cur_step >= min_step)
		{
			float new_ratio[4];
			float new_ab[4];

			new_ratio[0] = cur_ratio;
			new_ratio[1] = cur_ratio + cur_step;
			new_ratio[2] = cur_ratio;
			new_ratio[3] = cur_ratio - cur_step;

			new_ab[0] = cur_ab + cur_step;
			new_ab[1] = cur_ab;
			new_ab[2] = cur_ab - cur_step;
			new_ab[3] = cur_ab;

			for (int i = 0; i < 4; i++)
			{
				if (new_ratio[i] < 0.0f)
					new_ratio[i] = 0.0f;
				else if (new_ratio[i] > 1.0f)
					new_ratio[i] = 1.0f;

				if (new_ab[i] < 0.0f)
					new_ab[i] = 0.0f;
				else if (new_ab[i] > 1.0f)
					new_ab[i] = 1.0f;
			}

			float new_a[4];
			float new_b[4];

			int min_idx;
			float min_err;

			for (int i = 0; i < 4; i++)
			{
				new_a[i] = new_ab[i] * new_ratio[i];
				new_b[i] = new_ab[i] * (1.0f - new_ratio[i]);
			
				float cur_a = new_a[i];
				float cur_b = new_b[i];
				float cur_c = 1.0f - cur_a - cur_b;

				float dA_est = calphaa * cur_a + cbetaa * cur_b + cgammaa * cur_c;
				float dB_est = calphab * cur_a + cbetab * cur_b + cgammab * cur_c;

				float dA_err = pow(dA_est - dA, 2.0f);
				float dB_err = pow(dB_est - dB, 2.0f);

				float tot_err = dA_err + dB_err;

				if (i == 0 || tot_err < min_err)
				{
					min_idx = i;
					min_err = tot_err;
				}
			}

			if (min_err < cur_error)
			{
				cur_ratio = new_ratio[min_idx];
				cur_ab = new_ab[min_idx];
				cur_error = min_err;
			}
			else
			{
				cur_step = cur_step / 2.0f;
			}
		}

		float cur_best_a, cur_best_b, cur_best_c;

		switch(i)
		{
			case 0:
				cur_best_a = cur_ab * cur_ratio;
				cur_best_b = cur_ab * (1.0f - cur_ratio);
				cur_best_c = 1.0f - cur_ab;
				break;
			case 1:
				cur_best_c = cur_ab * cur_ratio;
				cur_best_a = cur_ab * (1.0f - cur_ratio);
				cur_best_b = 1.0f - cur_ab;
				break;
			case 2:
				cur_best_b = cur_ab * cur_ratio;
				cur_best_c = cur_ab * (1.0f - cur_ratio);
				cur_best_a = 1.0f - cur_ab;
				break;
		}

		tot_best_a += cur_best_a;
		tot_best_b += cur_best_b;
		tot_best_c += cur_best_c;
	}

	if(idx_adjust)
		idx = idx_adjust - idx;

	uchar best_a = (uchar)floor(tot_best_a / 3 * 255.0f);
	uchar best_b = (uchar)floor(tot_best_b / 3 * 255.0f);
	uchar best_c = (uchar)floor(tot_best_c / 3 * 255.0f);

	x[idx] = best_a;
	y[idx] = best_b;
	z[idx] = best_c;

	if(do_merge)
		m[idx] = (short)(dA * mr + dB * (1.0f - mr));
}
