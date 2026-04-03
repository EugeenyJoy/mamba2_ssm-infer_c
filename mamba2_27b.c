/*
 * MAMBA-2 2.7B INFERENCE ENGINE
 * For state-spaces/mamba2-2.7b weights
 *
 * gcc -O3 -march=native -o mamba2_27b mamba2_27b.c -lm
 * ./mamba2_27b mamba2_2.7b.bin "Hello world"
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* ============================================================
   CONFIG — read from binary header
   ============================================================ */
typedef struct {
    int d_model;     // 2560
    int n_layers;    // 64
    int n_heads;     // 80
    int head_dim;    // 64
    int d_state;     // 128
    int d_inner;     // 5120
    int n_groups;    // 1
    int d_conv;      // 4
    int vocab_size;  // 50288
} Config;

/* ============================================================
   HALF-FLOAT (fp16) support
   ============================================================ */
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

/* ============================================================
   LAYER WEIGHTS — stored as fp16 pointers
   ============================================================ */
typedef struct {
    uint16_t *norm_w;      // [d_model]
    uint16_t *in_proj_w;   // [d_inner*2 + n_groups*d_state*2 + n_heads, d_model]
    uint16_t *conv_w;      // [d_inner + n_groups*d_state*2, 1, d_conv]  
    uint16_t *conv_b;      // [d_inner + n_groups*d_state*2]
    uint16_t *dt_bias;     // [n_heads]
    uint16_t *A_log;       // [n_heads]
    uint16_t *D;           // [n_heads]
    uint16_t *norm_inner;  // [d_inner]
    uint16_t *out_proj_w;  // [d_model, d_inner]
} Layer;

typedef struct {
    Config cfg;
    uint16_t *embed;        // [vocab_size, d_model]
    Layer *layers;
    uint16_t *final_norm;   // [d_model]
    
    // Runtime state per layer
    float **ssm_state;      // [n_layers][n_heads * head_dim * d_state]
    float **conv_state;     // [n_layers][conv_dim * d_conv]
    
    // Buffers
    float *logits;
    
    // Raw mmap data
    uint16_t *raw_data;
} Model;

/* ============================================================
   MATH
   ============================================================ */
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float softplus(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

static void rmsnorm(float *out, const float *inp, const uint16_t *w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += inp[i] * inp[i];
    float scale = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++)
        out[i] = inp[i] * scale * fp16_to_fp32(w[i]);
}

/* Matrix multiply: out[out_dim] = w[out_dim, in_dim] @ inp[in_dim]
   w is fp16 */
static void matmul_fp16(float *out, const float *inp,
                        const uint16_t *w, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = 0;
        const uint16_t *row = w + (long)o * in_dim;
        for (int i = 0; i < in_dim; i++)
            sum += fp16_to_fp32(row[i]) * inp[i];
        out[o] = sum;
    }
}

/* ============================================================
   MODEL LOADING
   ============================================================ */
static Model* load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    
    // Check magic
    uint32_t magic;
    fread(&magic, 4, 1, f);
    if (magic != 0x4D414D42) {
        fprintf(stderr, "Bad magic: 0x%X (expected MAMB)\n", magic);
        fclose(f); return NULL;
    }
    
    Config cfg;
    fread(&cfg.d_model, 4, 1, f);
    fread(&cfg.n_layers, 4, 1, f);
    fread(&cfg.n_heads, 4, 1, f);
    fread(&cfg.head_dim, 4, 1, f);
    fread(&cfg.d_state, 4, 1, f);
    fread(&cfg.d_inner, 4, 1, f);
    fread(&cfg.n_groups, 4, 1, f);
    fread(&cfg.d_conv, 4, 1, f);
    fread(&cfg.vocab_size, 4, 1, f);
    
    printf("Config: d_model=%d layers=%d heads=%d head_dim=%d\n",
           cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.head_dim);
    printf("        d_state=%d d_inner=%d groups=%d conv=%d vocab=%d\n",
           cfg.d_state, cfg.d_inner, cfg.n_groups, cfg.d_conv, cfg.vocab_size);
    
    int D = cfg.d_model;
    int L = cfg.n_layers;
    int H = cfg.n_heads;
    int HD = cfg.head_dim;
    int S = cfg.d_state;
    int DI = cfg.d_inner;     // 5120
    int G = cfg.n_groups;      // 1
    int K = cfg.d_conv;        // 4
    int V = cfg.vocab_size;
    
    // in_proj output dim: z(DI) + x(DI) + B(G*S) + C(G*S) + dt(H)
    int in_proj_out = DI + DI + G*S + G*S + H;  // 10576
    // conv operates on: x(DI) + B(G*S) + C(G*S) 
    int conv_dim = DI + 2*G*S;  // 5376
    
    printf("in_proj_out=%d conv_dim=%d\n", in_proj_out, conv_dim);
    
    Model *m = (Model*)calloc(1, sizeof(Model));
    m->cfg = cfg;
    
    // Read embedding: [V, D] in fp16
    long embed_size = (long)V * D;
    m->embed = (uint16_t*)malloc(embed_size * 2);
    printf("Loading embedding (%ld MB)...\n", embed_size * 2 / 1024 / 1024);
    fread(m->embed, 2, embed_size, f);
    
    // Read layers
    m->layers = (Layer*)calloc(L, sizeof(Layer));
    
    for (int l = 0; l < L; l++) {
        if (l % 8 == 0) printf("Loading layer %d/%d...\n", l, L);
        Layer *ly = &m->layers[l];
        
        // norm: [D]
        ly->norm_w = (uint16_t*)malloc(D * 2);
        fread(ly->norm_w, 2, D, f);
        
        // in_proj: [in_proj_out, D]
        long ip_size = (long)in_proj_out * D;
        ly->in_proj_w = (uint16_t*)malloc(ip_size * 2);
        fread(ly->in_proj_w, 2, ip_size, f);
        
        // conv_w: [conv_dim, 1, K]
        long cw_size = (long)conv_dim * K;
        ly->conv_w = (uint16_t*)malloc(cw_size * 2);
        fread(ly->conv_w, 2, cw_size, f);
        
        // conv_b: [conv_dim]
        ly->conv_b = (uint16_t*)malloc(conv_dim * 2);
        fread(ly->conv_b, 2, conv_dim, f);
        
        // dt_bias: [H]
        ly->dt_bias = (uint16_t*)malloc(H * 2);
        fread(ly->dt_bias, 2, H, f);
        
        // A_log: [H]
        ly->A_log = (uint16_t*)malloc(H * 2);
        fread(ly->A_log, 2, H, f);
        
        // D: [H]
        ly->D = (uint16_t*)malloc(H * 2);
        fread(ly->D, 2, H, f);
        
        // norm (inner): [DI]
        ly->norm_inner = (uint16_t*)malloc(DI * 2);
        fread(ly->norm_inner, 2, DI, f);
        
        // out_proj: [D, DI]
        long op_size = (long)D * DI;
        ly->out_proj_w = (uint16_t*)malloc(op_size * 2);
        fread(ly->out_proj_w, 2, op_size, f);
    }
    
    // Final norm: [D]
    m->final_norm = (uint16_t*)malloc(D * 2);
    fread(m->final_norm, 2, D, f);
    
    fclose(f);
    
    // Allocate runtime state
    m->ssm_state = (float**)malloc(L * sizeof(float*));
    m->conv_state = (float**)malloc(L * sizeof(float*));
    for (int l = 0; l < L; l++) {
        // SSM state: [n_heads, head_dim, d_state]
        m->ssm_state[l] = (float*)calloc((long)H * HD * S, sizeof(float));
        // Conv state: [conv_dim, d_conv]
        m->conv_state[l] = (float*)calloc((long)conv_dim * K, sizeof(float));
    }
    
    m->logits = (float*)calloc(V, sizeof(float));
    
    // Memory usage
    long ip_total = (long)in_proj_out * D;
    long cw_total = (long)conv_dim * K;
    long per_layer = (long)D*2 + ip_total*2 + cw_total*2 + conv_dim*2 
                     + H*6 + DI*2 + (long)D*DI*2;
    long total = embed_size * 2 + per_layer * L + D * 2;
    long state_mem = L * ((long)H*HD*S*4 + (long)conv_dim*K*4) + V*4;
    printf("\nModel: %.0f MB weights + %.0f MB state\n",
           total/1024.0/1024, state_mem/1024.0/1024);
    printf("Model loaded!\n\n");
    
    return m;
}

/* ============================================================
   FORWARD — one token, Mamba-2 style
   ============================================================ */
static void forward_layer(Model *m, int l, float *hidden) {
    Config *c = &m->cfg;
    Layer *ly = &m->layers[l];
    int D = c->d_model;
    int H = c->n_heads;
    int HD = c->head_dim;
    int S = c->d_state;
    int DI = c->d_inner;     // H * HD = 80 * 64 = 5120
    int G = c->n_groups;
    int K = c->d_conv;
    
    int conv_dim = DI + 2*G*S;  // 5376
    int in_proj_out = DI + DI + 2*G*S + H;  // 10576
    
    // 1. RMSNorm
    float *normed = (float*)malloc(D * sizeof(float));
    rmsnorm(normed, hidden, ly->norm_w, D);
    
    // 2. in_proj: [in_proj_out] = W[in_proj_out, D] @ normed[D]
    float *proj = (float*)malloc(in_proj_out * sizeof(float));
    matmul_fp16(proj, normed, ly->in_proj_w, in_proj_out, D);
    free(normed);
    
    // Split proj into: z[DI], xBC[DI + 2*G*S], dt[H]
    float *z  = proj;                    // [0 .. DI-1]
    float *xBC = proj + DI;              // [DI .. DI+conv_dim-1]  
    float *dt = proj + DI + conv_dim;    // [DI+conv_dim .. end]
    
    // 3. Conv1d on xBC [conv_dim]
    float *conv_out = (float*)malloc(conv_dim * sizeof(float));
    float *cstate = m->conv_state[l];
    
    // Shift conv state and insert new
    for (int i = 0; i < conv_dim; i++) {
        for (int k = K-1; k > 0; k--)
            cstate[i*K + k] = cstate[i*K + k - 1];
        cstate[i*K + 0] = xBC[i];
    }
    
    // Apply conv: cross-correlation
    for (int i = 0; i < conv_dim; i++) {
        float sum = fp16_to_fp32(ly->conv_b[i]);
        for (int k = 0; k < K; k++)
            sum += fp16_to_fp32(ly->conv_w[i*K + k]) * cstate[i*K + (K-1-k)];
        conv_out[i] = silu(sum);
    }
    
    // Split conv_out: x[DI], B[G*S], C[G*S]
    float *x = conv_out;                     // [0..DI-1]
    float *B = conv_out + DI;                // [DI..DI+G*S-1]
    float *C = conv_out + DI + G*S;          // [DI+G*S..end]
    
    // 4. dt: add bias + softplus
    for (int h = 0; h < H; h++) {
        dt[h] = dt[h] + fp16_to_fp32(ly->dt_bias[h]);
        dt[h] = softplus(dt[h]);
    }
    
    // 5. SSM step — per head
    // State: [H, HD, S]
    // x reshaped as [H, HD]
    // B as [G, S] (G=1 so just [S])
    // C as [G, S]
    float *y = (float*)malloc(DI * sizeof(float));
    float *state = m->ssm_state[l];
    
    for (int h = 0; h < H; h++) {
        float A = -expf(fp16_to_fp32(ly->A_log[h]));
        float d = dt[h];
        float decay = expf(d * A);
        float D_val = fp16_to_fp32(ly->D[h]);
        
        int g = h / (H / G);  // group index (G=1 -> always 0)
        float *Bg = B + g * S;
        float *Cg = C + g * S;
        
        for (int hd = 0; hd < HD; hd++) {
            float x_val = x[h * HD + hd];
            float y_val = 0;
            
            for (int s = 0; s < S; s++) {
                long si = ((long)h * HD + hd) * S + s;
                state[si] = state[si] * decay + d * Bg[s] * x_val;
                y_val += state[si] * Cg[s];
            }
            
            y[h * HD + hd] = y_val + x_val * D_val;
        }
    }
    
    free(conv_out);
    
    // 6. RMSNorm on y (inner norm)
    float *y_normed = (float*)malloc(DI * sizeof(float));
    rmsnorm(y_normed, y, ly->norm_inner, DI);
    free(y);
    
    // 7. Gate: y = y_normed * silu(z)
    for (int i = 0; i < DI; i++)
        y_normed[i] *= silu(z[i]);
    
    // 8. out_proj: [D] = W[D, DI] @ y[DI]
    float *out = (float*)malloc(D * sizeof(float));
    matmul_fp16(out, y_normed, ly->out_proj_w, D, DI);
    free(y_normed);
    free(proj);
    
    // 9. Residual
    for (int i = 0; i < D; i++)
        hidden[i] += out[i];
    free(out);
}

static void forward_token(Model *m, int token, float *logits) {
    Config *c = &m->cfg;
    int D = c->d_model;
    int V = c->vocab_size;
    
    float *hidden = (float*)malloc(D * sizeof(float));
    
    // Embedding
    for (int i = 0; i < D; i++)
        hidden[i] = fp16_to_fp32(m->embed[(long)token * D + i]);
    
    // Layers
    for (int l = 0; l < c->n_layers; l++)
        forward_layer(m, l, hidden);
    
    // Final RMSNorm
    float *normed = (float*)malloc(D * sizeof(float));
    rmsnorm(normed, hidden, m->final_norm, D);
    free(hidden);
    
    // LM head (weight tying with embedding)
    matmul_fp16(logits, normed, m->embed, V, D);
    free(normed);
}

/* ============================================================
   RESET STATE
   ============================================================ */
static void reset_state(Model *m) {
    Config *c = &m->cfg;
    int conv_dim = c->d_inner + 2 * c->n_groups * c->d_state;
    for (int l = 0; l < c->n_layers; l++) {
        memset(m->ssm_state[l], 0, 
               (long)c->n_heads * c->head_dim * c->d_state * sizeof(float));
        memset(m->conv_state[l], 0,
               (long)conv_dim * c->d_conv * sizeof(float));
    }
}

/* ============================================================
   SAMPLING
   ============================================================ */
static int sample_token(float *logits, int V, float temp) {
    if (temp < 0.01f) {
        int best = 0;
        for (int i = 1; i < V; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
    
    float max_l = logits[0];
    for (int i = 1; i < V; i++)
        if (logits[i] > max_l) max_l = logits[i];
    
    float sum = 0;
    for (int i = 0; i < V; i++) {
        logits[i] = expf((logits[i] - max_l) / temp);
        sum += logits[i];
    }
    for (int i = 0; i < V; i++) logits[i] /= sum;
    
    float r = (float)rand() / RAND_MAX;
    float cum = 0;
    for (int i = 0; i < V; i++) {
        cum += logits[i];
        if (cum > r) return i;
    }
    return V - 1;
}

/* ============================================================
   BPE TOKENIZER (simple, load from file)
   ============================================================ */
// For now: just feed raw bytes through embedding
// TODO: proper BPE tokenizer

/* ============================================================
   MAIN
   ============================================================ */
int main(int argc, char **argv) {
    srand(time(NULL));
    
    if (argc < 2) {
        printf("Mamba-2 2.7B Inference\n");
        printf("  %s model.bin \"prompt\" [tokens] [temp]\n", argv[0]);
        return 1;
    }
    
    printf("Loading model: %s\n", argv[1]);
    Model *m = load_model(argv[1]);
    if (!m) return 1;
    
    const char *prompt = (argc >= 3) ? argv[2] : "The";
    int max_tokens = (argc >= 4) ? atoi(argv[3]) : 50;
    float temp = (argc >= 5) ? atof(argv[4]) : 0.7f;
    
    printf("Prompt: \"%s\" | Tokens: %d | Temp: %.2f\n\n", 
           prompt, max_tokens, temp);
    
    reset_state(m);
    
    // Feed prompt (as bytes for now — need BPE tokenizer)
    int last_tok = 0;
    int len = strlen(prompt);
    printf("\033[1;36m%s\033[0m", prompt);
    fflush(stdout);
    
    // TODO: BPE encode prompt
    // For now just feed ASCII bytes  
    for (int i = 0; i < len; i++) {
        last_tok = (unsigned char)prompt[i];
        forward_token(m, last_tok, m->logits);
    }
    
    // Generate
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    
    for (int i = 0; i < max_tokens; i++) {
        int tok = sample_token(m->logits, m->cfg.vocab_size, temp);
        // TODO: BPE decode token
        if (tok < 128) putchar(tok);
        else printf("[%d]", tok);
        fflush(stdout);
        forward_token(m, tok, m->logits);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + 
                     (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("\n\033[0;33m[%d tok, %.1fs, %.1f tok/s]\033[0m\n",
           max_tokens, elapsed, max_tokens/elapsed);
    
    return 0;
}
