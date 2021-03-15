#include "main.h"
#include "File.h"

#ifdef NO_SDL
#include "header.h"
namespace FileWAV {
    #include "FileNotImplemented.h"
}
#include "footer.h"
#else 

#include "SDL.h"
#include "Network.h" // for ntohs
#include "Geometry.h"

#include "header.h"
namespace FileWAV {
    void help() {
        printf(".wav sound files. They are represented as one or two channel images with\n"
               "height and width of 1, but many frames.\n");
    }

    void save(Window im, string filename) {
        FILE *f = fopen(filename.c_str(), "wb");
        assert(f, "Could not open %s\n", filename.c_str());
        assert(im.width == 1 && im.height == 1, 
               "wav files must have frames, but no width or height\n");
        assert(im.channels == 1 || im.channels == 2, 
               "wav files must have one or two channels\n");

        // 2 bytes per sample
        int dataSize = im.channels * im.frames * 2;
        unsigned long l;
        unsigned short s;

        // the riff header
        fprintf(f, "RIFF");
        l = 36 + dataSize;
        fwrite(&l, 4, 1, f);
        fprintf(f, "WAVE");

        // the format chunk
        fprintf(f, "fmt ");
        l = 16;
        fwrite(&l, 4, 1, f);

        s = 1;
        fwrite(&s, 2, 1, f);
        s = im.channels;
        fwrite(&s, 2, 1, f);
        l = 44100;
        fwrite(&l, 4, 1, f);
        l = 44100 * im.channels * 2;
        fwrite(&l, 4, 1, f);
        s = im.channels * 2;
        fwrite(&s, 2, 1, f);
        s = 16;
        fwrite(&s, 2, 1, f);

        fprintf(f, "data");
        l = dataSize;
        fwrite(&l, 4, 1, f);
        
        short maxVal = (short)0x7fff;
        short minVal = (short)0x8000;

        for (int t = 0; t < im.frames; t++) {
            for (int c = 0; c < im.channels; c++) {
                short sval;
                float fval = im(0, 0, t)[c];
                if (fval >= 1) sval = maxVal;
                else if (fval <= -1) sval = minVal;
                else sval = (short)(fval * maxVal + 0.499);
                fwrite(&sval, 2, 1, f);
            }
        }

        fclose(f);
    }

    Image load(string filename) {
        SDL_AudioSpec wav_spec;
        Uint32 wav_length;
        unsigned char *wav_buffer;
        assert(SDL_LoadWAV(filename.c_str(), &wav_spec, &wav_buffer, &wav_length),
               "Could not open %s\n", filename.c_str());


        int frames = wav_length / wav_spec.channels;
        switch(wav_spec.format) {
        case AUDIO_S8: case AUDIO_U8:
            break;
        default: // 16 bit formats
            frames /= 2;
            break;
        }
        Image sound(1, 1, frames, wav_spec.channels);

        switch(wav_spec.format) {
        case AUDIO_U8: {
            unsigned char *wavPtr = wav_buffer;
            float *sndPtr = sound(0, 0, 0);
            float mult = 1.0f/128;
            for (int t = 0; t < sound.frames; t++) {
                for (int c = 0; c < sound.channels; c++) {
                    *sndPtr++ = (*wavPtr++ - 128) * mult;
                }
            }
            break;
        }
        case AUDIO_S8: {
            char *wavPtr = (char *)wav_buffer;
            float *sndPtr = sound(0, 0, 0);
            float mult = 1.0f/128;
            for (int t = 0; t < sound.frames; t++) {
                for (int c = 0; c < sound.channels; c++) {
                    *sndPtr++ = *wavPtr++ * mult;
                }
            }
            break;
        }
        case AUDIO_S16MSB: {
            short *wavPtr = (short *)wav_buffer;
            float *sndPtr = sound(0, 0, 0);
            float mult = 1.0f/(1<<15);
            for (int t = 0; t < sound.frames; t++) {
                for (int c = 0; c < sound.channels; c++) {
                    *sndPtr++ = ntohs(*wavPtr++) * mult;
                }
            }
            break;
        }
        case AUDIO_U16MSB: {
            unsigned short *wavPtr = (unsigned short *)wav_buffer;
            float *sndPtr = sound(0, 0, 0);
            float mult = 1.0f/(1<<15);
            for (int t = 0; t < sound.frames; t++) {
                for (int c = 0; c < sound.channels; c++) {
                    *sndPtr++ = (ntohs(*wavPtr++)-(1<<15)) * mult;
                }
            }
            break;
        }
        case AUDIO_U16LSB: {
            unsigned short *wavPtr = (unsigned short *)wav_buffer;
            float *sndPtr = sound(0, 0, 0);
            float mult = 1.0f/(1<<15);
            for (int t = 0; t < sound.frames; t++) {
                for (int c = 0; c < sound.channels; c++) {
                    short val1 = ntohs(*wavPtr++);
                    short val2 = ((val1 & 255) << 8) | ((val1 >> 8) & 255);
                    *sndPtr++ = (val2 - (1<<15)) * mult;
                }
            }
            break;
        }
        case AUDIO_S16LSB: {
            short *wavPtr = (short *)wav_buffer;
            float *sndPtr = sound(0, 0, 0);
            float mult = 1.0f/(1<<15);
            for (int t = 0; t < sound.frames; t++) {
                for (int c = 0; c < sound.channels; c++) {
                    short val1 = ntohs(*wavPtr++);
                    short val2 = ((val1 & 255) << 8) | ((val1 >> 8) & 255);
                    *sndPtr++ = (val2) * mult;
                }
            }
            break;
        }
        default:
            panic("Unknown wav format!\n");
        }

        SDL_FreeWAV(wav_buffer);

        // resample to 44100
        if (wav_spec.freq != 44100) {
            return Resample::apply(sound, 1, 1, (sound.frames * 44100) / wav_spec.freq);
        } else {
            return sound;
        }
    }
}

#include "footer.h"
#endif

