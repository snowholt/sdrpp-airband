#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Minimal RIFF WAV reader/writer for the offline harness. Supports the two
// formats this project actually needs: 16-bit PCM and 32-bit IEEE float, any
// channel count folded to mono by averaging. No external dependencies, which
// keeps the harness buildable wherever SDR++ builds.
namespace ag {

    struct WavData {
        std::vector<float> samples;
        int rate = 0;
        bool ok = false;
        std::string err;
    };

    namespace wavdetail {

        inline uint16_t rdU16(const uint8_t* p) {
            return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
        }

        inline uint32_t rdU32(const uint8_t* p) {
            return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
                 | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        }

        inline void wrU16(std::vector<uint8_t>& out, uint16_t v) {
            out.push_back((uint8_t)(v & 0xFF));
            out.push_back((uint8_t)((v >> 8) & 0xFF));
        }

        inline void wrU32(std::vector<uint8_t>& out, uint32_t v) {
            out.push_back((uint8_t)(v & 0xFF));
            out.push_back((uint8_t)((v >> 8) & 0xFF));
            out.push_back((uint8_t)((v >> 16) & 0xFF));
            out.push_back((uint8_t)((v >> 24) & 0xFF));
        }

        inline void wrTag(std::vector<uint8_t>& out, const char* tag) {
            for (int i = 0; i < 4; i++) { out.push_back((uint8_t)tag[i]); }
        }

        inline float clampOne(float v) {
            if (!(v >= -1.0f)) { return -1.0f; }   // also catches NaN
            if (v > 1.0f) { return 1.0f; }
            return v;
        }

        inline bool writeWav(const char* path, const std::vector<float>& samples,
                             int rate, uint16_t audioFormat, uint16_t bitsPerSample) {
            if (!path || rate <= 0) { return false; }

            // PCM16 conversion first, so the payload size is known up front.
            std::vector<uint8_t> payload;
            if (audioFormat == 1 && bitsPerSample == 16) {
                payload.reserve(samples.size() * 2);
                for (float v : samples) {
                    float scaled = clampOne(v) * 32767.0f;
                    long r = std::lround(scaled);
                    if (r < -32768L) { r = -32768L; }
                    if (r > 32767L) { r = 32767L; }
                    wrU16(payload, (uint16_t)(int16_t)r);
                }
            }
            else if (audioFormat == 3 && bitsPerSample == 32) {
                payload.reserve(samples.size() * 4);
                for (float v : samples) {
                    uint32_t bits;
                    std::memcpy(&bits, &v, 4);
                    wrU32(payload, bits);
                }
            }
            else {
                return false;
            }

            std::vector<uint8_t> hdr;
            wrTag(hdr, "RIFF");
            wrU32(hdr, (uint32_t)(36 + payload.size()));
            wrTag(hdr, "WAVE");
            wrTag(hdr, "fmt ");
            wrU32(hdr, 16);                       // size of fmt chunk
            wrU16(hdr, audioFormat);
            wrU16(hdr, 1);                        // mono
            wrU32(hdr, (uint32_t)rate);
            wrU32(hdr, (uint32_t)(rate * bitsPerSample / 8));   // byte rate
            wrU16(hdr, (uint16_t)(bitsPerSample / 8));          // block align
            wrU16(hdr, bitsPerSample);
            wrTag(hdr, "data");
            wrU32(hdr, (uint32_t)payload.size());

            std::FILE* f = std::fopen(path, "wb");
            if (!f) { return false; }
            bool ok = true;
            ok = ok && (std::fwrite(hdr.data(), 1, hdr.size(), f) == hdr.size());
            if (!payload.empty()) {
                ok = ok && (std::fwrite(payload.data(), 1, payload.size(), f) == payload.size());
            }
            ok = (std::fclose(f) == 0) && ok;
            return ok;
        }
    }

    inline WavData read(const char* path) {
        WavData out;
        if (!path) { out.err = "null path"; return out; }

        std::FILE* f = std::fopen(path, "rb");
        if (!f) { out.err = "cannot open file"; return out; }
        std::fseek(f, 0, SEEK_END);
        long fileSize = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (fileSize < 12) { std::fclose(f); out.err = "file too small for RIFF"; return out; }

        std::vector<uint8_t> all((size_t)fileSize);
        if (std::fread(all.data(), 1, all.size(), f) != (size_t)fileSize) {
            std::fclose(f);
            out.err = "short read";
            return out;
        }
        std::fclose(f);

        if (std::memcmp(&all[0], "RIFF", 4) != 0 || std::memcmp(&all[8], "WAVE", 4) != 0) {
            out.err = "not a RIFF/WAVE file";
            return out;
        }

        uint16_t audioFormat = 0;
        uint16_t channels = 0;
        uint16_t bits = 0;
        bool haveFmt = false;
        bool haveData = false;
        size_t dataOff = 0;
        uint32_t dataSize = 0;

        size_t pos = 12;
        while (pos + 8 <= all.size()) {
            const char* id = (const char*)&all[pos];
            uint32_t size = wavdetail::rdU32(&all[pos + 4]);
            pos += 8;
            if (pos + size > all.size()) {
                // The chunk claims more payload than the file holds: truncated.
                out.err = "truncated chunk in file";
                return out;
            }
            if (!haveFmt && !std::memcmp(id, "fmt ", 4)) {
                audioFormat = wavdetail::rdU16(&all[pos]);
                channels = wavdetail::rdU16(&all[pos + 2]);
                out.rate = (int)wavdetail::rdU32(&all[pos + 4]);
                bits = wavdetail::rdU16(&all[pos + 14]);
                haveFmt = true;
            }
            else if (!haveData && !std::memcmp(id, "data", 4)) {
                dataOff = pos;
                dataSize = size;
                haveData = true;
            }
            // Chunks are word-aligned: an odd size carries one pad byte.
            pos += size + (size & 1u);
        }

        if (!haveFmt || !haveData) { out.err = "missing fmt or data chunk"; return out; }
        if (channels < 1 || channels > 2) { out.err = "unsupported channel count"; return out; }

        const bool pcm16 = (audioFormat == 1 && bits == 16);
        const bool f32 = (audioFormat == 3 && bits == 32);
        if (!pcm16 && !f32) { out.err = "unsupported sample format"; return out; }

        const size_t frameCount = dataSize / ((bits / 8) * channels);
        out.samples.resize(frameCount);
        const size_t frameBytes = (bits / 8) * channels;

        for (size_t fr = 0; fr < frameCount; fr++) {
            const uint8_t* p = &all[dataOff + fr * frameBytes];
            float acc = 0.0f;
            for (int c = 0; c < channels; c++) {
                const uint8_t* cp = p + c * (bits / 8);
                if (pcm16) {
                    acc += (float)(int16_t)wavdetail::rdU16(cp) / 32768.0f;
                }
                else {
                    uint32_t b = wavdetail::rdU32(cp);
                    float v;
                    std::memcpy(&v, &b, 4);
                    acc += v;
                }
            }
            out.samples[fr] = acc / (float)channels;
        }

        out.ok = true;
        return out;
    }

    inline bool writePcm16(const char* path, const std::vector<float>& samples, int rate) {
        return wavdetail::writeWav(path, samples, rate, 1, 16);
    }

    inline bool writeFloat32(const char* path, const std::vector<float>& samples, int rate) {
        return wavdetail::writeWav(path, samples, rate, 3, 32);
    }
}
