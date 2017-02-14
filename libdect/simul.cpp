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

/*
	(1)         a * alphaa + b * betaa + c * gammaa = theta
	(2)         a * alphab + b * betab + c * gammab = phi

	given c = 1 - a - b

	(1) -> (3)  a(alphaa - gammaa) + b(betaa - gammaa) + gammaa = theta
	(2) -> (4)  a(alphab - gammab) + b(betab - gammab) + gammab = phi

	let:
		alpha = alphaa - gammaa
		beta  = betaa - gammaa
		gamma = alphab - gammab
		delta = betab - gammab

	(3) -> (5)  a * alpha + b * beta = theta - gammaa
	(4) -> (6)  a * gamma + b * delta = phi - gammab

	(5) -> (7)  b = (theta - gammaa - a * alpha) / beta

	let:
	    epsilon = delta / beta

	(7) in (6)
	            a * gamma + epsilon * theta - epsilon * gamma - a * alpha * epsilon = phi - gammab
				a = (phi - gammab - epsilon(theta - gammaa)) / (gamma - alpha * epsilon)
*/

int dect_algo_simul(int enhanced,
	const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	uint8_t *x, uint8_t *y, uint8_t *z,
	size_t out_size,
	float min_step,
	int16_t *m,
	float mr,
	int idx_adjust)
{
	for (size_t idx = 0; idx < out_size; idx++)
	{
		float theta = a[idx];
		float phi = b[idx];

		float alpha = alphaa - gammaa;
		float beta = betaa - gammaa;
		float gamma = alphab - gammab;
		float delta = betab - gammab;

		float epsilon = delta / beta;
		float curx = (phi - gammab - epsilon * (theta - gammaa)) /
			(gamma - alpha * epsilon);
		float cury = (theta - gammaa - curx * alpha) / beta;
		float curz = 1.0f - curx - cury;

		auto out_idx = idx;
		if (idx_adjust)
			out_idx = idx_adjust - idx;

		int xo = (int)(curx * 255.0f);
		int yo = (int)(cury * 255.0f);
		int zo = (int)(curz * 255.0f);

		if (xo < 0)
			xo = 0;
		if (yo < 0)
			yo = 0;
		if (zo < 0)
			zo = 0;
		if (xo > 255)
			xo = 255;
		if (yo > 255)
			yo = 255;
		if (zo > 255)
			zo = 255;

		x[out_idx] = xo;
		y[out_idx] = yo;
		z[out_idx] = zo;

		if(m)
			m[out_idx] = (int16_t)(theta * mr + phi * (1.0f - mr));
	}

	return 0;
}
