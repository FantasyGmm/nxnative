// The same NVN sampler keeps its guest cap on a one-level subview and can use the override on a full mip chain.
// Observe actual Metal descriptors and replay the selected primary and border twin through the GPU argument block.
#import <Metal/Metal.h>
static id<MTLDevice> recording_device(void);
#define MTLCreateSystemDefaultDevice recording_device
#include "../nvn/nvn_mtl.m"
#undef MTLCreateSystemDefaultDevice

module_t *g_mod;
int g_nmod;
module_t *module_of(const void *p) { (void)p; return NULL; }
void *front_metal_layer(void) { return NULL; }
void svc_thread_dump(void) {}
uint64_t now_ns(void) { return bp_now(); }

static int checks, fails;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

@interface SamplerProbe : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) NSMutableArray<MTLSamplerDescriptor *> *descriptors;
@property(nonatomic) BOOL nativeReduction;
@end
@implementation SamplerProbe
- (id)forwardingTargetForSelector:(SEL)selector { (void)selector; return _device; }
- (id<MTLSamplerState>)newSamplerStateWithDescriptor:(MTLSamplerDescriptor *)descriptor {
    [_descriptors addObject:[descriptor copy]];   // samp_make reuses the same mutable descriptor for the twin
    // Metal validation rejects min/max reduction on pre-Apple10 GPUs. On those GPUs check the requested descriptor only.
    if (!_nativeReduction && descriptor.reductionMode != MTLSamplerReductionModeWeightedAverage) return nil;
    return [_device newSamplerStateWithDescriptor:descriptor];
}
@end

static SamplerProbe *probe;
static id<MTLDevice> recording_device(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return nil;
    probe = [SamplerProbe new]; probe.device = device; probe.descriptors = [NSMutableArray array];
    probe.nativeReduction = [device supportsFamily:MTLGPUFamilyApple10];
    return (id<MTLDevice>)probe;
}

enum { SINGLE_TID = 21, MIPPED_TID = 22, CS = 5 };

static id<MTLTexture> install_views(tex_t *guest) {
    *guest = (tex_t){.magic = MAGIC('T','E','X',0), .serial = 97000};
    guest->d = (nvn_texdesc){.target = NVN_TEXTURE_TARGET_2D, .format = NVN_FORMAT_RGBA8,
        .width = 4, .height = 4, .depth = 1, .levels = 3, .samples = 1};
    MTLTextureDescriptor *d = tex_desc(guest, 0);
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> base = [probe.device newTextureWithDescriptor:d];
    if (!base) return nil;
    for (NSUInteger l = 0, side = 4; l < base.mipmapLevelCount; l++, side >>= 1) {
        uint8_t pixels[4 * 4 * 4]; memset(pixels, 64 + 32 * l, sizeof pixels);
        [base replaceRegion:MTLRegionMake2D(0, 0, side, side) mipmapLevel:l withBytes:pixels bytesPerRow:4 * side];
    }
    res_add(base); mtl_tex_init(guest); serial_put(guest->serial, base);
    view_t v = {.set_levels = 1, .base_level = 1, .nlevels = 1};
    mtl_reg_tex(SINGLE_TID, guest, &v);
    mtl_reg_tex(MIPPED_TID, guest, NULL);
    return base;
}

static void gpu_pair(int sid, id<MTLTexture> single, id<MTLTexture> mipped, const uint64_t rids[2][2]) {
    id<MTLDevice> device = probe.device;
    NSError *error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:
        @"#include <metal_stdlib>\nusing namespace metal;\n"
        "struct A { sampler primary [[id(0)]]; sampler twin [[id(1)]]; sampler unpaired [[id(2)]];\n"
        "           texture2d<float> tex [[id(3)]]; };\n"
        "kernel void sample_pair(constant A &a [[buffer(0)]], device float *out [[buffer(1)]]) {\n"
        "    float2 inside(0.5f), outside(-2.f, 0.5f);\n"
        "    out[0] = a.tex.sample(a.primary, inside, level(0)).r;\n"
        "    out[1] = a.tex.sample(a.twin, inside, level(0)).r;\n"
        "    out[2] = a.tex.sample(a.primary, outside, level(0)).r;\n"
        "    out[3] = a.tex.sample(a.twin, outside, level(0)).r;\n"
        "    out[4] = a.tex.sample(a.unpaired, inside, level(0)).r;\n"
        "}\n" options:nil error:&error];
    id<MTLFunction> fn = [lib newFunctionWithName:@"sample_pair"];
    id<MTLComputePipelineState> pipeline = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
    CHECK(pipeline != nil, "sample_pair pipeline: %s", error ? error.localizedDescription.UTF8String : "missing function");
    if (!pipeline) return;

    NvShader *sh = [NvShader new]; sh->stage = CS; sh->fn = fn;
    sh->ae = [fn newArgumentEncoderWithBufferIndex:0]; sh->arg_len = sh->ae.encodedLength;
    sh->rf.nb = 4;
    // Samplers precede the texture in the block, so encode_args must defer both paired slots.
    sh->rf.b[0] = (nvn_rbind){.kind = NVN_RB_SAMPT, .aid = 0};
    sh->rf.b[1] = (nvn_rbind){.kind = NVN_RB_SAMPT, .aid = 1, .bw = 1};
    sh->rf.b[2] = (nvn_rbind){.kind = NVN_RB_SAMPT, .aid = 2};
    sh->rf.b[3] = (nvn_rbind){.kind = NVN_RB_TEX, .aid = 3, .dtt = MTLTextureType2D, .dpf = NVN_DPF_FLOAT};
    nvn_refl_pair_samplers(@{@"bindings": @[
        @{@"msl_name": @"texSmplr", @"msl_type": @"sampler"},
        @{@"msl_name": @"texSmplr" DMSH_BORDER_TWIN, @"msl_type": @"sampler"},
        @{@"msl_name": @"unpaired", @"msl_type": @"sampler"},
        @{@"msl_name": @"tex", @"msl_type": @"texture2d<float>"}
    ]}, &sh->rf);
    enum { STRIDE = 8, DISPATCHES = 4 };
    id<MTLBuffer> out = [device newBufferWithLength:DISPATCHES * STRIDE * sizeof(float) options:MTLResourceStorageModeShared];
    CHECK(out && sh->arg_len >= 4 * sizeof(uint64_t), "sample_pair argument and readback storage");
    if (!out || sh->arg_len < 4 * sizeof(uint64_t)) return;
    res_add(out);
    state_reset(); mtl_sub_begin();
    for (int i = 0; i < DISPATCHES; i++) {
        int separate = i == 3;
        int full = i == 1 || separate;
        uint64_t sampler = (uint64_t)sid << 20, texture = full ? MIPPED_TID : SINGLE_TID;
        p_bind_tex1 b = {.stage = CS, .h = sampler | texture};
        if (separate) {
            b.slot = NVN_SEPTEX_SLOT0 + 3; b.h = texture;
            p_bind_handles s = {.stage = CS, .first = 7, .n = 1, .h = {sampler}};
            state_op(OP_BIND_SAMP, &s);
            for (int j = 0; j < 3; j++) { sh->rf.b[j].kind = NVN_RB_SAMP; sh->rf.b[j].reg = s.first; }
            sh->rf.b[3].reg = b.slot;
        }
        state_op(OP_BIND_TEX1, &b);
        id<MTL4ComputeCommandEncoder> e = dispatch_enc(); enc_c_pso(e, pipeline);
        CHECK(encode_args((id<MTLCommandEncoder>)e, sh, CS, 0) && sh->ab_buf, "dispatch %d encodes the production argument block", i);
        if (!sh->ab_buf) break;
        const uint64_t *slots = (const uint64_t *)((const uint8_t *)sh->ab_buf.contents + sh->ab_off);
        CHECK(slots[0] == rids[full][0] && slots[1] == rids[full][1], "dispatch %d encodes the expected primary/twin resource IDs", i);
        CHECK(slots[2] == rids[0][0], "dispatch %d leaves the unpaired sampler at the guest cap", i);
        CHECK(slots[3] == (full ? mipped : single).gpuResourceID._impl, "dispatch %d encodes the selected Metal texture view", i);
        enc_c_args(e, out.gpuAddress + i * STRIDE * sizeof(float), 1);
        enc_c_dispatch(e, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
    }
    CHECK(mtl_wait(mtl_sub_commit(), 5000000000ull) && !mtl_last_commit_error(), "sample_pair submission completes");
    const float *pixels = out.contents;
    for (int i = 0; i < DISPATCHES; i++) {
        float inner = (i == 1 || i == 3 ? 64.f : 96.f) / 255.f;
        const float want[] = {inner, inner, 0, 1, inner};
        for (int j = 0; j < 5; j++) {
            float got = pixels[i * STRIDE + j];
            CHECK(fabsf(got - want[j]) < 1e-3f, "dispatch %d sample %d: %g, expected %g", i, j, got, want[j]);
        }
    }
}

typedef struct {
    const char *name;
    NVNminFilter min;
    MTLSamplerMipFilter mip;
    float guest_aniso;
    NSUInteger guest_max;
    bool eligible, nearest_min, nearest_mag, compare, border;
    NVNsamplerReduction reduction;
    bool lod_clamped;
} SamplerCase;

static const SamplerCase cases[] = {
    {"linear, nearest mip", NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST, MTLSamplerMipFilterNearest, 0, 1, .eligible = true},
    {"linear, linear mip", NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR, MTLSamplerMipFilterLinear, 32, 16, .eligible = true},
    {"custom border", NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR, MTLSamplerMipFilterLinear, 4, 4, .eligible = true, .border = true},
    {"nearest min", NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR, MTLSamplerMipFilterLinear, 1, 1, .nearest_min = true},
    {"nearest mag", NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST, MTLSamplerMipFilterNearest, 1, 1, .nearest_mag = true},
    {"no mip", NVN_MIN_FILTER_LINEAR, MTLSamplerMipFilterNotMipmapped, 1, 1},
    {"comparison", NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR, MTLSamplerMipFilterLinear, 1, 1, .compare = true},
    {"LOD clamp", NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR, MTLSamplerMipFilterLinear, 1, 1, .lod_clamped = true},
    {"min reduction", NVN_MIN_FILTER_LINEAR, MTLSamplerMipFilterLinear, 1, 1, .reduction = NVN_SAMPLER_REDUCTION_MIN},
    {"max reduction", NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST, MTLSamplerMipFilterLinear, 1, 1, .reduction = NVN_SAMPLER_REDUCTION_MAX},
};

static void registered_case(int sid, const SamplerCase *c, NSUInteger cap, id<MTLTexture> single, id<MTLTexture> mipped) {
    samp_par p = {
        .minf = c->min, .magf = c->nearest_mag ? NVN_MAG_FILTER_NEAREST : NVN_MAG_FILTER_LINEAR,
        .ws = c->border ? NVN_WRAP_MODE_CLAMP_TO_BORDER : NVN_WRAP_MODE_REPEAT,
        .wt = NVN_WRAP_MODE_MIRRORED_REPEAT, .wr = NVN_WRAP_MODE_CLAMP_TO_EDGE,
        .cmp_mode = c->compare ? NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE : NVN_COMPARE_MODE_NONE,
        .cmp_func = NVN_COMPARE_FUNC_LEQUAL, .reduction = c->reduction,
        .lod_min = c->lod_clamped ? 0.25f : 0.f, .lod_max = 3.5f, .lod_bias = -0.75f, .aniso = c->guest_aniso,
    };
    if (c->border) memcpy(p.border, (float[4]){0.5f, 0.5f, 0.5f, 0.5f}, sizeof p.border);
    [probe.descriptors removeAllObjects];
    mtl_reg_samp(sid, &p);
    CHECK(probe.descriptors.count == 0, "%s: registration stays lazy", c->name);
    bool expect_state = c->reduction == NVN_SAMPLER_REDUCTION_AVERAGE || probe.nativeReduction;
    bool expect_variant = cap && c->eligible && cap != c->guest_max;
    int count = c->border ? 2 : 1;
    uint64_t unpaired_rid = 0, rids[2][2] = {{0}};
    id<MTLSamplerState> selected[2][2] = {{nil}};
    os_unfair_lock_lock(&g_rmu);
    id<MTLSamplerState> unpaired = expect_state ? samp_pick(sid, 0, 0, &unpaired_rid, NULL) : nil;
    for (int full = expect_state ? 0 : 1; full < 2; full++) {
        int aniso = g_sampler_anisotropy && samp_aniso_view(full ? mipped : single);
        for (int twin = 0; twin < count; twin++) {
            int correct = 0;
            selected[full][twin] = samp_pick(sid, aniso, twin, &rids[full][twin], &correct);
            CHECK(correct == twin, "%s: %s view, border correction follows twin", c->name, full ? "mipped" : "single-mip");
        }
    }
    os_unfair_lock_unlock(&g_rmu);

    if (expect_state) {
        CHECK(unpaired && selected[0][0] == unpaired && rids[0][0] == unpaired_rid,
              "%s: single-mip and unpaired bindings share the guest sampler", c->name);
        for (int twin = 0; twin < count; twin++) {
            CHECK(selected[0][twin] && rids[0][twin] && selected[1][twin] && rids[1][twin] &&
                  (rids[0][twin] != rids[1][twin]) == expect_variant,
                  "%s: mipped view %s the guest %s", c->name, expect_variant ? "replaces" : "keeps", twin ? "twin" : "primary");
        }
        if (c->border) CHECK(rids[0][0] != rids[0][1] && rids[1][0] != rids[1][1], "%s: primary and twin are distinct", c->name);
    } else {
        CHECK(!selected[1][0], "%s: this GPU cannot create the reduction sampler", c->name);
    }
    NSUInteger descriptors = count * (expect_variant ? 2 : 1);
    CHECK(probe.descriptors.count == descriptors, "%s: descriptor count %lu, expected %lu", c->name,
          (unsigned long)probe.descriptors.count, (unsigned long)descriptors);
    float lod_min = c->reduction == NVN_SAMPLER_REDUCTION_MIN ? 0 : p.lod_min;
    float lod_max = c->reduction == NVN_SAMPLER_REDUCTION_MIN ? 0 : 3.5f;
    MTLSamplerReductionMode reduction = c->reduction == NVN_SAMPLER_REDUCTION_MIN ? MTLSamplerReductionModeMinimum
        : c->reduction == NVN_SAMPLER_REDUCTION_MAX ? MTLSamplerReductionModeMaximum : MTLSamplerReductionModeWeightedAverage;
    NSUInteger index = 0;
    for (MTLSamplerDescriptor *d in probe.descriptors) {
        bool twin = index % count != 0;
        const char *which = twin ? "twin" : "primary";
        bool variant = index++ >= (NSUInteger)count;
        NSUInteger want = variant ? cap : c->guest_max;
        CHECK(d.maxAnisotropy == want, "%s %s %s: maxAnisotropy %lu, expected %lu", c->name, variant ? "override" : "guest",
              which, (unsigned long)d.maxAnisotropy, (unsigned long)want);
        CHECK(d.minFilter == (c->nearest_min ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear) &&
              d.magFilter == (c->nearest_mag ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear) &&
              d.mipFilter == c->mip, "%s %s: min/mag/mip filters", c->name, which);
        CHECK(d.sAddressMode == (c->border ? MTLSamplerAddressModeClampToBorderColor : MTLSamplerAddressModeRepeat) &&
              d.tAddressMode == MTLSamplerAddressModeMirrorRepeat && d.rAddressMode == MTLSamplerAddressModeClampToEdge,
              "%s %s: address modes", c->name, which);
        CHECK(d.borderColor == (twin ? MTLSamplerBorderColorOpaqueWhite : MTLSamplerBorderColorTransparentBlack),
              "%s %s: border color", c->name, which);
        CHECK(d.lodMinClamp == lod_min && d.lodMaxClamp == lod_max && d.lodBias == -0.75f, "%s %s: LOD clamp and bias", c->name, which);
        CHECK(d.compareFunction == (c->compare ? MTLCompareFunctionLessEqual : MTLCompareFunctionNever) && d.reductionMode == reduction,
              "%s %s: comparison and reduction", c->name, which);
        CHECK(d.supportArgumentBuffers && d.normalizedCoordinates, "%s %s: argument buffer and coordinate mode", c->name, which);
    }
    if (c->border && selected[0][0] && selected[0][1] && selected[1][0] && selected[1][1]) {
        gpu_pair(sid, single, mipped, rids);
        CHECK(probe.descriptors.count == descriptors, "%s: repeated binds reuse both sampler variants", c->name);
    }
}

static void pool_zero_case(void) {
    [probe.descriptors removeAllObjects];
    os_unfair_lock_lock(&g_rmu);
    uint64_t rid = 0;
    id<MTLSamplerState> s = samp_pick(SAMP_POOL_ZERO, g_sampler_anisotropy != 0, 0, &rid, NULL);
    os_unfair_lock_unlock(&g_rmu);
    MTLSamplerDescriptor *d = probe.descriptors.firstObject;
    CHECK(s && rid && probe.descriptors.count == 1 && d.maxAnisotropy == 1, "reserved pool sampler keeps anisotropy 1");
    CHECK(d.minFilter == MTLSamplerMinMagFilterLinear && d.magFilter == MTLSamplerMinMagFilterLinear &&
          d.mipFilter == MTLSamplerMipFilterNotMipmapped && d.lodMaxClamp == 15 && d.supportArgumentBuffers &&
          d.sAddressMode == MTLSamplerAddressModeClampToEdge && d.tAddressMode == MTLSamplerAddressModeClampToEdge &&
          d.rAddressMode == MTLSamplerAddressModeClampToEdge, "reserved pool sampler keeps its filters, clamps and binding modes");
}

int main(int argc, char **argv) { @autoreleasepool {
    NSUInteger cap = argc > 1 ? (NSUInteger)atoi(argv[1]) : 0;   // independently supplied by the Makefile
    const char *env = getenv("DM_NVN_ANISOTROPY");
    NSString *startup = env ? @(env) : @"<unset>";
    mtl_init();
    CHECK(setenv("DM_NVN_ANISOTROPY", cap ? "0" : "16", 1) == 0, "change the setting after initialization");
    mtl_init();   // Re-enter initialization before any sampler lookup; the startup setting must still win.
    CHECK(g_sampler_anisotropy == cap, "startup cap %lu, expected %lu", (unsigned long)g_sampler_anisotropy, (unsigned long)cap);
    pool_zero_case();
    tex_t guest;
    id<MTLTexture> base = install_views(&guest);
    os_unfair_lock_lock(&g_rmu);
    id<MTLTexture> single = (__bridge id<MTLTexture>)tex_fast(SINGLE_TID);
    id<MTLTexture> mipped = (__bridge id<MTLTexture>)tex_fast(MIPPED_TID);
    os_unfair_lock_unlock(&g_rmu);
    CHECK(base && mipped == base && mipped.mipmapLevelCount == 3 && single.parentTexture == base &&
          single.parentRelativeLevel == 1 && single.mipmapLevelCount == 1, "register a full texture and a one-mip view of the same storage");
    if (single && mipped)
        for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) registered_case(40 + (int)i, &cases[i], cap, single, mipped);
    printf("t_anisotropy (DM_NVN_ANISOTROPY='%s'): %d checks, %d failed\n", startup.UTF8String, checks, fails);
    return fails != 0;
} }
