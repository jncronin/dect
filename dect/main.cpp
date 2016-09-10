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

#include <cstdio>
#include <tiffio.h>

#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#include <tchar.h>
#include "XGetopt.h"

int dect_algo_opencl(const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	uint8_t *x, uint8_t *y, uint8_t *z,
	size_t outsize,
	float min_step,
	int16_t *m,
	float mr,
	int idx_adjust);

int opencl_init(int platform);
int opencl_dump_platforms();

static int(*dect_algo)(
	const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	uint8_t *x, uint8_t *y, uint8_t *z,
	size_t outsize,
	float min_step,
	int16_t *m,
	float mr,
	int idx_adjust);

#define DEF_ALPHAA 52.0f
#define DEF_BETAA -995.0f
#define DEF_GAMMAA 525.0f
#define DEF_ALPHAB 53.0f
#define DEF_BETAB -994.0f
#define DEF_GAMMAB 200.0f
#define DEF_MINSTEP 0.001f
#define DEF_MERGEFACT 0.5f

static float alphaa = DEF_ALPHAA;
static float betaa = DEF_BETAA;
static float gammaa = DEF_GAMMAA;
static float alphab = DEF_ALPHAB;
static float betab = DEF_BETAB;
static float gammab = DEF_GAMMAB;
static float min_step = DEF_MINSTEP;
int enhanced = 0;
static float merge_fact = DEF_MERGEFACT;

static int16_t *readTIFFDirectory(TIFF *f, size_t *buf_size)
{
	tdata_t buf;
	tstrip_t strip;
	buf = _TIFFmalloc(TIFFStripSize(f));
	auto size = TIFFStripSize(f);
	for (strip = 0; strip < TIFFNumberOfStrips(f); strip++)
		TIFFReadEncodedStrip(f, strip, buf, (tsize_t)-1);

	int16_t *ret = (int16_t *)buf;
	for (auto i = 0; i < size / 2; i++)
	{
		if (ret[i] >= 32768)
			ret[i] = ret[i] - 65536;
	}

	*buf_size = (size_t)size / 2;

	return ret;
}

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
static void dect_algo_cpu(const int16_t *a, const int16_t *b,
	float alphaa, float betaa, float gammaa,
	float alphab, float betab, float gammab,
	int idx,
	uint8_t *x, uint8_t *y, uint8_t *z,
	float min_step,
	int16_t *m,
	float mr,
	int idx_adjust)
{
	float dA = a[idx];
	float dB = b[idx];

	float tot_best_a = 0.0f;
	float tot_best_b = 0.0f;
	float tot_best_c = 0.0f;

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
		float cur_ratio = 0.5f;
		float cur_ab = 0.66f;
		float cur_step = 0.25f;
		float cur_error = 5000.0f * 5000.0f;

		float calphaa, cbetaa, cgammaa;
		float calphab, cbetab, cgammab;

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

				float dA_err = pow(dA_est - dA, 2);
				float dB_err = pow(dB_est - dB, 2);

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

		switch (i)
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

	if (enhanced)
	{
		tot_best_a /= enhanced;
		tot_best_b /= enhanced;
		tot_best_c /= enhanced;
	}

	if (idx_adjust)
		idx = idx_adjust - idx;

	uint8_t best_a = (uint8_t)floor(tot_best_a * 255.0f);
	uint8_t best_b = (uint8_t)floor(tot_best_b * 255.0f);
	uint8_t best_c = (uint8_t)floor(tot_best_c * 255.0f);

	x[idx] = best_a;
	y[idx] = best_b;
	z[idx] = best_c;

	if (m)
		m[idx] = (uint16_t)(dA * mr + dB * (1.0f - mr));
}

static int dect_algo_cpu_iter(
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
		dect_algo_cpu(a, b, alphaa, betaa, gammaa,
			alphab, betab, gammab, i, x, y, z, min_step,
			m, mr, idx_adjust);
	return 0;
}

/* Convert TCHAR* to UTF-8 for passing to libtiff */
static char *ascii(const TCHAR *s)
{
#ifdef _UNICODE
	size_t size;
	wcstombs_s(&size, NULL, 0, s, 0);
	char *ret = (char *)(malloc(size));
	wcstombs_s(&size, ret, size, s, _TRUNCATE);
	return ret;
#else
	return s;
#endif
}

static void help(TCHAR *fname)
{
	std::cout << "Usage:" << std::endl;
	std::cout << fname << " -A file_A.tiff -B file_B.tiff [options]" << std::endl;
	std::cout << std::endl;
	std::cout << "Options:" << std::endl;
	std::cout << " -x file             output for material a (defaults to outputx.tiff)" << std::endl;
	std::cout << " -y file             output for material b (defaults to outputy.tiff)" << std::endl;
	std::cout << " -z file             output for material c (defaults to outputz.tiff)" << std::endl;
	std::cout << " -a density          density for material a in file A (defaults to " << DEF_ALPHAA << ")" << std::endl;
	std::cout << " -b density          density for material b in file A (defaults to " << DEF_BETAA << ")" << std::endl;
	std::cout << " -c density          density for material c in file A (defaults to " << DEF_GAMMAA << ")" << std::endl;
	std::cout << " -d density          density for material a in file B (defaults to " << DEF_ALPHAB << ")" << std::endl;
	std::cout << " -e density          density for material b in file B (defaults to " << DEF_BETAB << ")" << std::endl;
	std::cout << " -f density          density for material c in file B (defaults to " << DEF_GAMMAB << ")" << std::endl;
	std::cout << " -m min_step         step size at which to stop searching (defaults to " << DEF_MINSTEP << ")" << std::endl;
	std::cout << " -D device_number    device to use for calculations (defaults to 0 i.e. CPU)" << std::endl;
	std::cout << " -E                  even bias for materials - slower" << std::endl;
	std::cout << " -M file             generate a merged image file too" << std::endl;
	std::cout << " -r ratio            ratio of A:B to use for merged image (defaults to " << DEF_MERGEFACT << ")" << std::endl;
	std::cout << " -F                  rotate outputted images 180 degrees" << std::endl;
	std::cout << " -h                  display this help" << std::endl;
	std::cout << std::endl;
	std::cout << "Devices" << std::endl;
	std::cout << " 0: CPU" << std::endl;
	opencl_dump_platforms();
	std::cout << std::endl;
}

int _tmain(int argc, TCHAR *argv[])
{
	size_t a_len, b_len;
	dect_algo = dect_algo_cpu_iter;

	TCHAR *afname = NULL;
	TCHAR *bfname = NULL;
	TCHAR *xfname = _T("outputx.tiff");
	TCHAR *yfname = _T("outputy.tiff");
	TCHAR *zfname = _T("outputz.tiff");
	TCHAR *mfname = NULL;
	int dev = 0;
	int do_rotate = 0;

	int g;
	while ((g = getopt(argc, argv, _T("A:B:x:y:z:D:a:b:c:d:e:f:g:hm:EM:r:F"))) != -1)
	{
		switch (g)
		{
		case 'A':
			afname = optarg;
			break;
		case 'B':
			bfname = optarg;
			break;

		case 'x':
			xfname = optarg;
			break;
		case 'y':
			yfname = optarg;
			break;
		case 'z':
			zfname = optarg;
			break;

		case 'D':
			dev = _ttoi(optarg);
			break;

		case 'a':
			alphaa = (float)_ttof(optarg);
			break;
		case 'b':
			betaa = (float)_ttof(optarg);
			break;
		case 'c':
			gammaa = (float)_ttof(optarg);
			break;
		case 'd':
			alphab = (float)_ttof(optarg);
			break;
		case 'e':
			betab = (float)_ttof(optarg);
			break;
		case 'f':
			gammab = (float)_ttof(optarg);
			break;

		case 'm':
			min_step = (float)_ttof(optarg);
			break;

		case 'h':
			help(argv[0]);
			return 0;

		case 'E':
			enhanced = 3;
			break;

		case 'M':
			mfname = optarg;
			break;

		case 'r':
			merge_fact = (float)_ttof(optarg);
			break;

		case 'F':
			do_rotate = 1;
			break;

		default:
			std::cout << "Unknown argument: " << (char)g << std::endl;
			help(argv[0]);
			return 0;
		}
	}

	if (afname == NULL || bfname == NULL)
	{
		help(argv[0]);
		return 0;
	}

	auto atest = ascii(afname);
	
	auto af = TIFFOpen(ascii(afname), "r");
	auto bf = TIFFOpen(ascii(bfname), "r");

	auto cf = TIFFOpen(ascii(xfname), "w");
	auto df = TIFFOpen(ascii(yfname), "w");
	auto ef = TIFFOpen(ascii(zfname), "w");

	TIFF *mf = NULL;
	if (mfname)
		mf = TIFFOpen(ascii(mfname), "w");

	assert(af);
	assert(bf);
	assert(cf);
	assert(df);
	assert(ef);

	if (dev == 0)
		dect_algo = dect_algo_cpu_iter;
	else
	{
		if (opencl_init(dev - 1) != 0)
		{
			std::cerr << "ERROR: opencl_init() failed, switching to CPU" << std::endl;
			dect_algo = dect_algo_cpu_iter;
		}
		else
			dect_algo = dect_algo_opencl;
	}


	int frame_id = 0;

	do
	{
		auto a = readTIFFDirectory(af, &a_len);
		auto b = readTIFFDirectory(bf, &b_len);

		assert(a);
		assert(b);
		assert(a_len == b_len);

		uint8_t *x = (uint8_t *)malloc(a_len);
		uint8_t *y = (uint8_t *)malloc(a_len);
		uint8_t *z = (uint8_t *)malloc(a_len);

		int16_t *m = NULL;
		if (mf)
			m = (int16_t *)malloc(a_len * 2);

		// run the algorithm
		auto algo_ret = dect_algo(a, b, alphaa, betaa, gammaa,
			alphab, betab, gammab,
			x, y, z, a_len, min_step, m, merge_fact,
			do_rotate ? (a_len - 1) : 0);
		if (algo_ret != 0)
		{
			if (dect_algo == dect_algo_cpu_iter)
			{
				std::cerr << "ERROR: CPU algorithm failed" << std::endl;
				exit(0);
			}
			else if (dect_algo == dect_algo_opencl)
			{
				std::cerr << "ERROR: OpenCL algorithm failed, switching to CPU" << std::endl;
				dect_algo = dect_algo_cpu_iter;
				algo_ret = dect_algo(a, b,
					alphaa, betaa, gammaa,
					alphab, betab, gammab,
					x, y, z, a_len, min_step, m, merge_fact,
					do_rotate ? (a_len  - 1): 0);
				if (algo_ret != 0)
				{
					std::cerr << "ERROR: CPU algorithm also failed" << std::endl;
					exit(0);
				}
			}
			else
			{
				std::cerr << "ERROR: unknown algorithm failed" << std::endl;
				exit(0);
			}
		}

		// attempt to write something out
		uint32 iw, il, rps;
		uint16 o, comp;
		uint16 spp = 1;
		uint16 bps = 8;
		uint16 pc = PLANARCONFIG_CONTIG;
		uint16 ru, ph;
		float xp = 0.0f, yp = 0.0f, xr, yr;
		int ret;
		ret = TIFFGetField(af, TIFFTAG_IMAGEWIDTH, &iw);
		assert(ret == 1);
		ret = TIFFGetField(af, TIFFTAG_IMAGELENGTH, &il);
		assert(ret == 1);
		ret = TIFFGetField(af, TIFFTAG_ORIENTATION, &o);
		assert(ret == 1);
		ret = TIFFGetField(af, TIFFTAG_ROWSPERSTRIP, &rps);
		assert(ret == 1);
		ret = TIFFGetField(af, TIFFTAG_COMPRESSION, &comp);
		assert(ret == 1);
		ret = TIFFGetField(af, TIFFTAG_RESOLUTIONUNIT, &ru);
		assert(ret == 1);
		ret = TIFFGetField(af, TIFFTAG_PHOTOMETRIC, &ph);
		assert(ret == 1);
		if (TIFFGetField(af, TIFFTAG_XPOSITION, &xp) != 1)
			xp = 0;
		if (TIFFGetField(af, TIFFTAG_YPOSITION, &yp) != 1)
			yp = 0;
		ret = TIFFGetField(af, TIFFTAG_XRESOLUTION, &xr);
		assert(ret == 1);
		ret = TIFFGetField(af, TIFFTAG_YRESOLUTION, &yr);
		assert(ret == 1);


		ret = TIFFSetField(cf, TIFFTAG_IMAGEWIDTH, iw);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_IMAGELENGTH, il);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_SAMPLESPERPIXEL, spp);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_BITSPERSAMPLE, bps);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_ORIENTATION, o);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_PLANARCONFIG, pc);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_ROWSPERSTRIP, rps);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_COMPRESSION, comp);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_RESOLUTIONUNIT, ru);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_PHOTOMETRIC, ph);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_XPOSITION, xp);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_YPOSITION, yp);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_XRESOLUTION, xr);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_YRESOLUTION, yr);
		assert(ret == 1);
		ret = TIFFSetField(cf, TIFFTAG_SAMPLEFORMAT, 1);
		assert(ret == 1);

		ret = TIFFSetField(df, TIFFTAG_IMAGEWIDTH, iw);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_IMAGELENGTH, il);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_SAMPLESPERPIXEL, spp);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_BITSPERSAMPLE, bps);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_ORIENTATION, o);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_PLANARCONFIG, pc);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_ROWSPERSTRIP, rps);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_COMPRESSION, comp);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_RESOLUTIONUNIT, ru);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_PHOTOMETRIC, ph);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_XPOSITION, xp);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_YPOSITION, yp);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_XRESOLUTION, xr);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_YRESOLUTION, yr);
		assert(ret == 1);
		ret = TIFFSetField(df, TIFFTAG_SAMPLEFORMAT, 1);
		assert(ret == 1);

		ret = TIFFSetField(ef, TIFFTAG_IMAGEWIDTH, iw);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_IMAGELENGTH, il);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_SAMPLESPERPIXEL, spp);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_BITSPERSAMPLE, bps);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_ORIENTATION, o);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_PLANARCONFIG, pc);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_ROWSPERSTRIP, rps);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_COMPRESSION, comp);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_RESOLUTIONUNIT, ru);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_PHOTOMETRIC, ph);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_XPOSITION, xp);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_YPOSITION, yp);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_XRESOLUTION, xr);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_YRESOLUTION, yr);
		assert(ret == 1);
		ret = TIFFSetField(ef, TIFFTAG_SAMPLEFORMAT, 1);
		assert(ret == 1);

		TIFFWriteEncodedStrip(cf, 0, x, a_len);
		TIFFWriteDirectory(cf);

		TIFFWriteEncodedStrip(df, 0, y, a_len);
		TIFFWriteDirectory(df);

		TIFFWriteEncodedStrip(ef, 0, z, a_len);
		TIFFWriteDirectory(ef);

		_TIFFfree(a);
		_TIFFfree(b);

		free(x);
		free(y);
		free(z);

		if (m)
		{
			ret = TIFFSetField(mf, TIFFTAG_IMAGEWIDTH, iw);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_IMAGELENGTH, il);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_SAMPLESPERPIXEL, 1);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_BITSPERSAMPLE, 16);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_ORIENTATION, o);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_PLANARCONFIG, pc);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_ROWSPERSTRIP, rps);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_COMPRESSION, comp);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_RESOLUTIONUNIT, ru);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_PHOTOMETRIC, ph);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_XPOSITION, xp);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_YPOSITION, yp);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_XRESOLUTION, xr);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_YRESOLUTION, yr);
			assert(ret == 1);
			ret = TIFFSetField(mf, TIFFTAG_SAMPLEFORMAT, 2);
			assert(ret == 1);

			TIFFWriteEncodedStrip(mf, 0, m, a_len * 2);
			TIFFWriteDirectory(mf);

			free(m);
		}

		printf("Processed frame %i\n", frame_id++);
	} while (TIFFReadDirectory(af) && TIFFReadDirectory(bf));

	TIFFFlush(cf);
	TIFFClose(cf);
	TIFFFlush(df);
	TIFFClose(df);
	TIFFFlush(ef);
	TIFFClose(ef);

	TIFFClose(af);
	TIFFClose(bf);

	return 0;
}
