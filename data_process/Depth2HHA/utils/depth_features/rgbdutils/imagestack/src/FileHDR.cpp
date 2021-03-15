#include "main.h"
#include "File.h"

#include "header.h"
namespace FileHDR {

    void help() {
        printf(".hdr files. These always have three channels and one frame. They store data\n"
               "in a 4 bytes per pixel red green blue exponent (rgbe) format.\n");
    }
  
#define  RED                0
#define  GRN                1
#define  BLU                2
#define  EXP                3
#define  COLXS                128        /* excess used for exponent */

    typedef unsigned char  BYTE;        /* 8-bit unsigned integer */

    typedef BYTE  COLR[4];                /* red, green, blue, exponent */

#define  copycolr(c1,c2)        (c1[0]=c2[0],c1[1]=c2[1],        \
                                 c1[2]=c2[2],c1[3]=c2[3])

    typedef float  COLOR[3];        /* red, green, blue */

#define  colval(col,pri)        ((col)[pri])

#define  setcolor(col,r,g,b)        ((col)[RED]=(r),(col)[GRN]=(g),(col)[BLU]=(b))

#define  copycolor(c1,c2)        ((c1)[0]=(c2)[0],(c1)[1]=(c2)[1],(c1)[2]=(c2)[2])

#define  scalecolor(col,sf)        ((col)[0]*=(sf),(col)[1]*=(sf),(col)[2]*=(sf))

#define  addcolor(c1,c2)        ((c1)[0]+=(c2)[0],(c1)[1]+=(c2)[1],(c1)[2]+=(c2)[2])

#define  multcolor(c1,c2)        ((c1)[0]*=(c2)[0],(c1)[1]*=(c2)[1],(c1)[2]*=(c2)[2])

#ifdef  NTSC
#define  bright(col)                (.295*(col)[RED]+.636*(col)[GRN]+.070*(col)[BLU])
#define  normbright(c)                (int)((74L*(c)[RED]+164L*(c)[GRN]+18L*(c)[BLU])/256)
#else
#define  bright(col)                (.263*(col)[RED]+.655*(col)[GRN]+.082*(col)[BLU])
#define  normbright(c)                (int)((67L*(c)[RED]+168L*(c)[GRN]+21L*(c)[BLU])/256)
#endif

#define  intens(col)                ( (col)[0] > (col)[1]                        \
                                  ? (col)[0] > (col)[2] ? (col)[0] : (col)[2] \
                                  : (col)[1] > (col)[2] ? (col)[1] : (col)[2] )

#define  colrval(c,p)                ( (c)[EXP] ?                                \
                                  ldexp((c)[p]+.5,(int)(c)[EXP]-(COLXS+8)) : \
                                  0. )

#define  WHTCOLOR                {1.0,1.0,1.0}
#define  BLKCOLOR                {0.0,0.0,0.0}
#define  WHTCOLR                {128,128,128,COLXS+1}
#define  BLKCOLR                {0,0,0,0}

    /* definitions for resolution header */
#define  XDECR                        1
#define  YDECR                        2
#define  YMAJOR                        4

    /* picture format identifier */
#define  COLRFMT                "32-bit_rle_rgbe"

    /* macros for exposures */
#define  EXPOSSTR                "EXPOSURE="
#define  LEXPOSSTR                9
#define  isexpos(hl)                (!strncmp(hl,EXPOSSTR,LEXPOSSTR))
#define  exposval(hl)                atof((hl)+LEXPOSSTR)
#define  fputexpos(ex,fp)        fprintf(fp,"%s%e\n",EXPOSSTR,ex)

    /* macros for pixel aspect ratios */
#define  ASPECTSTR                "PIXASPECT="
#define  LASPECTSTR                10
#define  isaspect(hl)                (!strncmp(hl,ASPECTSTR,LASPECTSTR))
#define  aspectval(hl)                atof((hl)+LASPECTSTR)
#define  fputaspect(pa,fp)        fprintf(fp,"%s%f\n",ASPECTSTR,pa)

    /* macros for color correction */
#define  COLCORSTR                "COLORCORR="
#define  LCOLCORSTR                10
#define  iscolcor(hl)                (!strncmp(hl,COLCORSTR,LCOLCORSTR))
#define  colcorval(cc,hl)        sscanf(hl+LCOLCORSTR,"%f %f %f",        \
                                       &(cc)[RED],&(cc)[GRN],&(cc)[BLU])
#define  fputcolcor(cc,fp)        fprintf(fp,"%s %f %f %f\n",COLCORSTR,        \
                                        (cc)[RED],(cc)[GRN],(cc)[BLU])


    /* Copyright (c) 1991 Regents of the University of California */

    /*
     *  color.c - routines for color calculations.
     *
     *     10/10/85
     */

#define  MINELEN        8        /* minimum scanline length for encoding */
#define  MINRUN                4        /* minimum run length */

    char * tempbuffer(size_t len)                        /* get a temporary buffer */
    {
        static char  *tempbuf = NULL;
        static size_t tempbuflen = 0;

        if (len > tempbuflen) {
            if (tempbuflen > 0)
                tempbuf = (char *)realloc((void *)tempbuf, len);
            else
                tempbuf = (char *)malloc(len);
            tempbuflen = tempbuf==NULL ? 0 : len;
        }
        return(tempbuf);
    }


    int fwritecolrs(COLR *scanline, int len, FILE *fp)  /* write out a colr scanline */
    {
        int  i, j, beg, cnt = 0;
        int  c2;
        
                if (len < MINELEN) {                /* too small to encode */
            int written = (int)(fwrite((char *)scanline,sizeof(COLR),len,fp)) - len;
                        return written;
                } if (len > 32767) {                /* too big! */
            return -1;
                }
        putc(2, fp);                        /* put magic header */
        putc(2, fp);
        putc(len>>8, fp);
        putc(len&255, fp);
        /* put components separately */
        for (i = 0; i < 4; i++) {
            for (j = 0; j < len; j += cnt) {        /* find next run */
                for (beg = j; beg < len; beg += cnt) {
                    for (cnt = 1; cnt < 127 && beg+cnt < len &&
                             scanline[beg+cnt][i] == scanline[beg][i]; cnt++)
                        ;
                    if (cnt >= MINRUN)
                        break;                        /* long enough */
                }
                if (beg-j > 1 && beg-j < MINRUN) {
                    c2 = j+1;
                    while (scanline[c2++][i] == scanline[j][i])
                        if (c2 == beg) {        /* short run */
                            putc(128+beg-j, fp);
                            putc(scanline[j][i], fp);
                            j = beg;
                            break;
                        }
                }
                while (j < beg) {                /* write out non-run */
                    if ((c2 = beg-j) > 128) c2 = 128;
                    putc(c2, fp);
                    while (c2--)
                        putc(scanline[j++][i], fp);
                }
                if (cnt >= MINRUN) {                /* write out run */
                    putc(128+cnt, fp);
                    putc(scanline[beg][i], fp);
                } else
                    cnt = 0;
            }
        }
        return(ferror(fp) ? -1 : 0);
    }

    int oldreadcolrs(COLR *scanline, int len, FILE *fp)                /* read in an old colr scanline */
    {
        int  rshift;
        int  i;
        
        rshift = 0;
        
        while (len > 0) {
            scanline[0][RED] = getc(fp);
            scanline[0][GRN] = getc(fp);
            scanline[0][BLU] = getc(fp);
            scanline[0][EXP] = getc(fp);
            if (feof(fp) || ferror(fp))
                return(-1);
            if (scanline[0][RED] == 1 &&
                scanline[0][GRN] == 1 &&
                scanline[0][BLU] == 1) {
                for (i = scanline[0][EXP] << rshift; i > 0; i--) {
                    copycolr(scanline[0], scanline[-1]);
                    scanline++;
                    len--;
                }
                rshift += 8;
            } else {
                scanline++;
                len--;
                rshift = 0;
            }
        }
        return(0);
    }

    int freadcolrs(COLR *scanline, int len, FILE *fp)  /* read in an encoded colr scanline */
    {
        int  i, j;
        int  code;
        /* determine scanline type */
        if (len < MINELEN)
            return(oldreadcolrs(scanline, len, fp));
        if ((i = getc(fp)) == EOF)
            return(-1);
        if (i != 2) {
            ungetc(i, fp);
            return(oldreadcolrs(scanline, len, fp));
        }
        scanline[0][GRN] = getc(fp);
        scanline[0][BLU] = getc(fp);
        if ((i = getc(fp)) == EOF)
            return(-1);
        if (scanline[0][GRN] != 2 || scanline[0][BLU] & 128) {
            scanline[0][RED] = 2;
            scanline[0][EXP] = i;
            return(oldreadcolrs(scanline+1, len-1, fp));
        }
        if ((scanline[0][BLU]<<8 | i) != len)
            return(-1);                /* length mismatch! */
        /* read each component */
        for (i = 0; i < 4; i++)
            for (j = 0; j < len; ) {
                if ((code = getc(fp)) == EOF)
                    return(-1);
                if (code > 128) {        /* run */
                    scanline[j++][i] = getc(fp);
                    for (code &= 127; --code; j++)
                        scanline[j][i] = scanline[j-1][i];
                } else                        /* non-run */
                    while (code--)
                        scanline[j++][i] = getc(fp);
            }
        return(feof(fp) ? -1 : 0);
    }

    void setcolr(COLR clr, double r, double g, double b)                /* assign a short color value */
    {
        double  d;
        int  e;
        
        d = r > g ? r : g;
        if (b > d) d = b;

        if (d <= 1e-32) {
            clr[RED] = clr[GRN] = clr[BLU] = 0;
            clr[EXP] = 0;
            return;
        }

        d = frexp(d, &e) * 256.0 / d;

        clr[RED] = (unsigned char)(r * d);
        clr[GRN] = (unsigned char)(g * d);
        clr[BLU] = (unsigned char)(b * d);
        clr[EXP] = (unsigned char)(e + COLXS);
    }

    int fwritescan(COLOR *scanline, int len, FILE *fp)                /* write out a scanline */
    {
        COLR *clrscan;
        int  n;
        COLR *sp;
        /* get scanline buffer */
        if ((sp = (COLR *)tempbuffer(len*sizeof(COLR))) == NULL)
            return(-1);
        clrscan = sp;
        /* convert scanline */
        n = len;
        while (n-- > 0) {
            setcolr(sp[0], scanline[0][RED],
                    scanline[0][GRN],
                    scanline[0][BLU]);
            scanline++;
            sp++;
        }
        return(fwritecolrs(clrscan, len, fp));
    }


    void colr_color(COLOR col, COLR clr)                /* convert short to float color */
    {
        double  f;
        
        if (clr[EXP] == 0)
            col[RED] = col[GRN] = col[BLU] = 0.0;
        else {
            f = ldexp(1.0, (int)clr[EXP]-(COLXS+8));
            col[RED] = (clr[RED] + 0.5f)*f;
            col[GRN] = (clr[GRN] + 0.5f)*f;
            col[BLU] = (clr[BLU] + 0.5f)*f;
        }
    }


    int freadscan(COLOR *scanline, int len, FILE *fp)                /* read in a scanline */
    {
        COLR  *clrscan;

        if ((clrscan = (COLR *)tempbuffer(len*sizeof(COLR))) == NULL)
            return(-1);
        if (freadcolrs(clrscan, len, fp) < 0)
            return(-1);
        /* convert scanline */
        colr_color(scanline[0], clrscan[0]);
        while (--len > 0) {
            scanline++; clrscan++;
            if (clrscan[0][RED] == clrscan[-1][RED] &&
                clrscan[0][GRN] == clrscan[-1][GRN] &&
                clrscan[0][BLU] == clrscan[-1][BLU] &&
                clrscan[0][EXP] == clrscan[-1][EXP])
                copycolor(scanline[0], scanline[-1]);
            else
                colr_color(scanline[0], clrscan[0]);
        }
        return(0);
    }


    int bigdiff(COLOR c1, COLOR c2, double md)                        /* c1 delta c2 > md? */
    {
        int  i;

        for (i = 0; i < 3; i++)
            if (colval(c1,i)-colval(c2,i) > md*colval(c2,i) ||
                colval(c2,i)-colval(c1,i) > md*colval(c1,i))
                return(1);
        return(0);
    }


    void save(Window im, string filename) {
                
        assert(im.channels == 3, "Can't save HDR image with <> 3 channels.\n");
        assert(im.frames == 1, "Can't save a multi-frame HDR image\n");
    
        FILE *f = fopen(filename.c_str(), "wb");            
        
        assert(f, "Could not write output file %s\n", filename.c_str());
        
        // Write trivial hdr header
        fprintf(f,"#?RADIANCE\n");
        fprintf(f,"\n");
        fprintf(f,"-Y %d +X %d\n", im.height, im.width);
        
        // Read image
        for (int y = 0; y < im.height; y++) {
            fwritescan((COLOR *)im(0, y), im.width, f);
        }
        
        fclose(f);
    }

    Image load(string filename) {
        FILE *f = fopen(filename.c_str(), "rb");            
        
        assert(f, "Could not open file\n");
        
        // Skip hdr header
        char thisChar = fgetc(f), lastChar;
        do {
            lastChar = thisChar;
            thisChar = fgetc(f);
        } while        (lastChar != '\n' || thisChar != '\n');            
        
        int height, width;
        assert(2 == fscanf(f, "-Y %d +X %d\n", &height, &width), 
               "Could not parse HDR header\n");
        Image im(width, height, 1, 3);
        
        // Read image
        for (int y = 0; y < height; y++) {
            FileHDR::freadscan((COLOR *)im(0, y), width, f);
        }
        
        fclose(f);

        return im;
    }
    
}
#include "footer.h"
