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

#ifdef _MSC_VER
#include <tchar.h>
#include "XGetopt.h"
#else
#define _T
#define TCHAR char
#define _ttoi atoi
#define _ttof atof
#define _tmain main
#include <getopt.h>
#endif

#include <libdect.h>

int dect_algo;

/* Values from http://xrayphysics.com/dual_energy.html,
	tissues are soft tissue, air and iodine */
#define DEF_ALPHAA 62.0f
#define DEF_BETAA -1000.0f
#define DEF_GAMMAA 512.0f
#define DEF_ALPHAB 58.0f
#define DEF_BETAB -1000.0f
#define DEF_GAMMAB 397.0f

#define DEF_MINSTEP 0.001f
#define DEF_MERGEFACT 0.5f

static float alphaa = DEF_ALPHAA;
static float betaa = DEF_BETAA;
static float gammaa = DEF_GAMMAA;
static float alphab = DEF_ALPHAB;
static float betab = DEF_BETAB;
static float gammab = DEF_GAMMAB;
static float min_step = DEF_MINSTEP;
int enhanced = 1;
static float merge_fact = DEF_MERGEFACT;
static int quiet = 0;

static int16_t *readTIFFDirectory(TIFF *f, size_t *buf_size)
{
	char *buf;
	tstrip_t strip;
	auto ssize = TIFFStripSize(f);
	auto size = ssize * TIFFNumberOfStrips(f);
	buf = (char *)_TIFFmalloc(size);
	for (strip = 0; strip < TIFFNumberOfStrips(f); strip++)
	{
		auto read = TIFFReadEncodedStrip(f, strip, &buf[strip * ssize], ssize);
		assert(read == ssize);
	}

	int sf = SAMPLEFORMAT_UINT;
	TIFFGetField(f, TIFFTAG_SAMPLEFORMAT, &sf);

	int16_t *ret = (int16_t *)buf;

	// handle different sample formats
	switch(sf)
	{
	case SAMPLEFORMAT_UINT:
		for (unsigned int i = 0; i < size / 2; i++)
			ret[i] = (int16_t)((int32_t)((uint16_t *)buf)[i] - 32768);
		break;
	}

	*buf_size = (size_t)size / 2;

	return ret;
}

static uint8_t *readTIFFDirectory2(TIFF *f, size_t *buf_size)
{
	tdata_t buf;
	tstrip_t strip;
	buf = _TIFFmalloc(TIFFStripSize(f));
	auto size = TIFFStripSize(f);
	for (strip = 0; strip < TIFFNumberOfStrips(f); strip++)
		TIFFReadEncodedStrip(f, strip, buf, (tsize_t)-1);

	uint8_t *ret = (uint8_t *)buf;

	*buf_size = (size_t)size;

	return ret;
}

/* Convert TCHAR* to UTF-8 for passing to libtiff */
char *ascii(const TCHAR *s)
{
#ifdef _UNICODE
	size_t size;
	wcstombs_s(&size, NULL, 0, s, 0);
	char *ret = (char *)(malloc(size));
	wcstombs_s(&size, ret, size, s, _TRUNCATE);
	return ret;
#else
	return (char *)s;
#endif
}

static void help(TCHAR *fname)
{
	std::cout << "Usage:" << std::endl;
	std::cout << ascii(fname) << " -A file_A.tiff -B file_B.tiff [options]" << std::endl;
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
	std::cout << " -F                  rotate output images 180 degrees" << std::endl;
	std::cout << " -q                  suppress progress output" << std::endl;
	std::cout << " -R                  reconstitute source images (overwrites source)" << std::endl;
	std::cout << " -h                  display this help" << std::endl;
	std::cout << std::endl;
	std::cout << "Devices" << std::endl;

	auto dev_count = dect_getDeviceCount();
	for (auto i = 0; i < dev_count; i++)
		std::cout << " " << i << ": " << dect_getDeviceName(i) << std::endl;
	std::cout << std::endl;
}

int _tmain(int argc, TCHAR *argv[])
{
	size_t a_len, b_len;
	dect_algo = 0;

	TCHAR *afname = NULL;
	TCHAR *bfname = NULL;
	TCHAR xfnamedef[] = _T("outputx.tiff");
	TCHAR yfnamedef[] = _T("outputy.tiff");
	TCHAR zfnamedef[] = _T("outputz.tiff");
	TCHAR *xfname = xfnamedef;
	TCHAR *yfname = yfnamedef;
	TCHAR *zfname = zfnamedef;
	TCHAR *mfname = NULL;
	int do_rotate = 0;
	int reconstitute = 0;

	int g;
	while ((g = getopt(argc, argv, _T("qA:B:x:y:z:D:a:b:c:d:e:f:g:hm:EM:r:FZR"))) != -1)
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
			dect_algo = _ttoi(optarg);
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

		case 'q':
			quiet = 1;
			break;

		case 'Z':
			quiet = 2;
			break;

		case 'R':
			reconstitute = 1;
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

	if (reconstitute)
	{
		auto af = TIFFOpen(ascii(afname), "w");
		auto bf = TIFFOpen(ascii(bfname), "w");

		auto cf = TIFFOpen(ascii(xfname), "r");
		auto df = TIFFOpen(ascii(yfname), "r");
		auto ef = TIFFOpen(ascii(zfname), "r");

		assert(af);
		assert(bf);
		assert(cf);
		assert(df);
		assert(ef);

		size_t c_len, d_len, e_len;

		do

		{
			auto c = readTIFFDirectory2(cf, &c_len);
			auto d = readTIFFDirectory2(df, &d_len);
			auto e = readTIFFDirectory2(ef, &e_len);

			assert(c);
			assert(d);
			assert(e);
			assert(c_len == d_len);
			assert(d_len == e_len);

			int16_t *a = (int16_t *)malloc(c_len * 2);
			int16_t *b = (int16_t *)malloc(c_len * 2);

			dect_reconstitute(c, d, e,
				alphaa, betaa, gammaa,
				alphab, betab, gammab,
				a, b, c_len,
				do_rotate ? ((int)c_len - 1) : 0);

			// attempt to write something out
			uint32 iw, il, rps;
			uint16 o, comp;
			uint16 spp = 1;
			uint16 bps = 16;
			uint16 pc = PLANARCONFIG_CONTIG;
			uint16 ru, ph;
			float xp = 0.0f, yp = 0.0f, xr, yr;
			int ret;
			ret = TIFFGetField(cf, TIFFTAG_IMAGEWIDTH, &iw);
			assert(ret == 1);
			ret = TIFFGetField(cf, TIFFTAG_IMAGELENGTH, &il);
			assert(ret == 1);
			ret = TIFFGetField(cf, TIFFTAG_ORIENTATION, &o);
			assert(ret == 1);
			ret = TIFFGetField(cf, TIFFTAG_ROWSPERSTRIP, &rps);
			assert(ret == 1);
			ret = TIFFGetField(cf, TIFFTAG_COMPRESSION, &comp);
			assert(ret == 1);
			ret = TIFFGetField(cf, TIFFTAG_RESOLUTIONUNIT, &ru);
			assert(ret == 1);
			ret = TIFFGetField(cf, TIFFTAG_PHOTOMETRIC, &ph);
			assert(ret == 1);
			if (TIFFGetField(cf, TIFFTAG_XPOSITION, &xp) != 1)
				xp = 0;
			if (TIFFGetField(cf, TIFFTAG_YPOSITION, &yp) != 1)
				yp = 0;
			ret = TIFFGetField(cf, TIFFTAG_XRESOLUTION, &xr);
			assert(ret == 1);
			ret = TIFFGetField(cf, TIFFTAG_YRESOLUTION, &yr);
			assert(ret == 1);


			ret = TIFFSetField(af, TIFFTAG_IMAGEWIDTH, iw);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_IMAGELENGTH, il);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_SAMPLESPERPIXEL, spp);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_BITSPERSAMPLE, bps);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_ORIENTATION, o);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_PLANARCONFIG, pc);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_ROWSPERSTRIP, rps);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_COMPRESSION, comp);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_RESOLUTIONUNIT, ru);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_PHOTOMETRIC, ph);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_XPOSITION, xp);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_YPOSITION, yp);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_XRESOLUTION, xr);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_YRESOLUTION, yr);
			assert(ret == 1);
			ret = TIFFSetField(af, TIFFTAG_SAMPLEFORMAT, 2);
			assert(ret == 1);

			ret = TIFFSetField(bf, TIFFTAG_IMAGEWIDTH, iw);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_IMAGELENGTH, il);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_SAMPLESPERPIXEL, spp);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_BITSPERSAMPLE, bps);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_ORIENTATION, o);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_PLANARCONFIG, pc);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_ROWSPERSTRIP, rps);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_COMPRESSION, comp);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_RESOLUTIONUNIT, ru);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_PHOTOMETRIC, ph);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_XPOSITION, xp);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_YPOSITION, yp);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_XRESOLUTION, xr);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_YRESOLUTION, yr);
			assert(ret == 1);
			ret = TIFFSetField(bf, TIFFTAG_SAMPLEFORMAT, 2);
			assert(ret == 1);

			TIFFWriteEncodedStrip(af, 0, a, (tsize_t)c_len * 2);
			TIFFWriteDirectory(af);

			TIFFWriteEncodedStrip(bf, 0, b, (tsize_t)c_len * 2);
			TIFFWriteDirectory(bf);

			_TIFFfree(c);
			_TIFFfree(d);
			_TIFFfree(e);

			free(a);
			free(b);
		} while (TIFFReadDirectory(cf) && TIFFReadDirectory(df) && TIFFReadDirectory(ef));

		TIFFFlush(af);
		TIFFClose(af);
		TIFFFlush(bf);
		TIFFClose(bf);

		TIFFClose(cf);
		TIFFClose(df);
		TIFFClose(ef);
	}
	else
	{
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

		int frame_id = 0;

		dect_initDevice(dect_algo, enhanced);

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
			auto algo_ret = dect_process(
				dect_algo, enhanced,
				a, b, alphaa, betaa, gammaa,
				alphab, betab, gammab,
				x, y, z, a_len, min_step, m, merge_fact,
				do_rotate ? ((int)a_len - 1) : 0);
			if (algo_ret != 0)
			{
				std::cerr << "ERROR: DECT algorithm failed" << std::endl;
				exit(0);
			}

			// attempt to write something out
			uint32 iw, il, rps;
			uint16 o = ORIENTATION_TOPLEFT, comp = COMPRESSION_LZW;
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
			//assert(ret == 1);
			ret = TIFFGetField(af, TIFFTAG_ROWSPERSTRIP, &rps);
			assert(ret == 1);
			//ret = TIFFGetField(af, TIFFTAG_COMPRESSION, &comp);
			//assert(ret == 1);
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
			//ret = TIFFSetField(cf, TIFFTAG_ROWSPERSTRIP, rps);
			//assert(ret == 1);
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
			//ret = TIFFSetField(df, TIFFTAG_ROWSPERSTRIP, rps);
			//assert(ret == 1);
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
			//ret = TIFFSetField(ef, TIFFTAG_ROWSPERSTRIP, rps);
			//assert(ret == 1);
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

			TIFFWriteEncodedStrip(cf, 0, x, (tsize_t)a_len);
			TIFFWriteDirectory(cf);

			TIFFWriteEncodedStrip(df, 0, y, (tsize_t)a_len);
			TIFFWriteDirectory(df);

			TIFFWriteEncodedStrip(ef, 0, z, (tsize_t)a_len);
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

				TIFFWriteEncodedStrip(mf, 0, m, (tsize_t)a_len * 2);
				TIFFWriteDirectory(mf);

				free(m);
			}

			switch (quiet)
			{
			case 0:
				printf("Processed frame %i\n", frame_id++);
				break;
			case 2:
				printf(".\n");
				break;
			}
		} while (TIFFReadDirectory(af) && TIFFReadDirectory(bf));

		TIFFFlush(cf);
		TIFFClose(cf);
		TIFFFlush(df);
		TIFFClose(df);
		TIFFFlush(ef);
		TIFFClose(ef);

		if (mf)
		{
			TIFFFlush(mf);
			TIFFClose(mf);
		}

		TIFFClose(af);
		TIFFClose(bf);
	}

	return 0;
}
