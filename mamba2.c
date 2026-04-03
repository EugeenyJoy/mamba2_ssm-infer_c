/*
 * MAMBA-2 INFERENCE ENGINE v3
 * Мультиязычный: Русский + English + Code
 * Интерактивный чат
 *
 * gcc -O3 -march=native -o mamba2 mamba2.c -lm
 *
 * Режимы:
 *   ./mamba2 model.bin "Привет" 200 0.7          — генерация
 *   ./mamba2 model.bin --chat                     — интерактивный чат
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================
   STRUCTURES
   ============================================================ */

typedef struct
{
    int vocab_size, d_model, n_layers, d_state, d_ff, chunk_size;
} Config;

typedef struct
{
    float *data;
    int ndim, shape[4], size;
} Tensor;

typedef struct
{
    /* Per SSM layer */
    Tensor in_w;           /* d_model → d_inner*2 */
    Tensor conv_w, conv_b; /* depthwise conv1d    */
    Tensor dt_w, dt_b;     /* dt projection       */
    Tensor B_w, C_w;       /* B,C projections     */
    Tensor log_A, D_param; /* A,D parameters      */
    Tensor out_w;          /* d_inner → d_model   */
    Tensor norm_w, norm_b; /* layernorm           */
} SSMLayer;

typedef struct
{
    /* Per FFN layer */
    Tensor up_w, up_b;
    Tensor down_w, down_b;
    Tensor norm_w, norm_b;
} FFNLayer;

typedef struct
{
    Config cfg;
    Tensor embed; /* vocab × d_model    */
    SSMLayer *ssm;
    FFNLayer *ffn;
    Tensor final_norm_w, final_norm_b;
    /* head = embed (weight tying) */

    /* Buffers */
    float **ssm_state; /* [layer][d_inner * d_state] */
    float *conv_buf;   /* circular conv buffer       */
    float *x_buf;      /* intermediate buffers       */
    float *z_buf;
    float *dt_buf;
    float *B_buf;
    float *C_buf;
    float *h_buf;
    float *logits;
} Model;

/* ============================================================
   MATH HELPERS
   ============================================================ */

static inline float silu(float x)
{
    return x / (1.0f + expf(-x));
}

static inline float gelu(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void layernorm(float *out, const float *inp, const float *w,
                      const float *b, int n)
{
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++)
        mean += inp[i];
    mean /= n;
    for (int i = 0; i < n; i++)
    {
        float d = inp[i] - mean;
        var += d * d;
    }
    var = 1.0f / sqrtf(var / n + 1e-5f);
    for (int i = 0; i < n; i++)
        out[i] = (inp[i] - mean) * var * w[i] + b[i];
}

static void matmul(float *out, const float *inp,
                   const float *w, int out_dim, int in_dim)
{
    for (int o = 0; o < out_dim; o++)
    {
        float sum = 0;
        const float *row = w + o * in_dim;
        for (int i = 0; i < in_dim; i++)
            sum += row[i] * inp[i];
        out[o] = sum;
    }
}

static void matmul_bias(float *out, const float *inp,
                        const float *w, const float *bias,
                        int out_dim, int in_dim)
{
    for (int o = 0; o < out_dim; o++)
    {
        float sum = bias[o];
        const float *row = w + o * in_dim;
        for (int i = 0; i < in_dim; i++)
            sum += row[i] * inp[i];
        out[o] = sum;
    }
}

static float softplus(float x)
{
    if (x > 20.0f)
        return x;
    return logf(1.0f + expf(x));
}

/* ============================================================
   MODEL LOADING
   ============================================================ */

static unsigned int read_u32(FILE *f)
{
    unsigned int v;
    fread(&v, 4, 1, f);
    return v;
}

static Tensor load_tensor_data(float *data, int ndim, int *shape)
{
    Tensor t;
    t.data = data;
    t.ndim = ndim;
    t.size = 1;
    for (int i = 0; i < ndim; i++)
    {
        t.shape[i] = shape[i];
        t.size *= shape[i];
    }
    return t;
}

typedef struct
{
    char name[256];
    Tensor t;
} NamedTensor;

static int find_tensor(NamedTensor *tensors, int n, const char *name)
{
    for (int i = 0; i < n; i++)
    {
        if (strcmp(tensors[i].name, name) == 0)
            return i;
    }
    return -1;
}

static Tensor get_tensor(NamedTensor *tensors, int n, const char *name)
{
    int i = find_tensor(tensors, n, name);
    if (i < 0)
    {
        fprintf(stderr, "Tensor not found: %s\n", name);
        Tensor empty = {NULL, 0, {0}, 0};
        return empty;
    }
    return tensors[i].t;
}

static Model *load_model(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        fprintf(stderr, "Cannot open %s\n", path);
        return NULL;
    }

    /* Config JSON */
    unsigned int json_len = read_u32(f);
    char *json = (char *)malloc(json_len + 1);
    fread(json, 1, json_len, f);
    json[json_len] = 0;

    Config cfg = {256, 256, 8, 16, 512, 64};
    /* Parse simple JSON */
    char *p;
    if ((p = strstr(json, "\"d_model\"")))
        sscanf(p, "%*[^:]:%d", &cfg.d_model);
    if ((p = strstr(json, "\"n_layers\"")))
        sscanf(p, "%*[^:]:%d", &cfg.n_layers);
    if ((p = strstr(json, "\"d_state\"")))
        sscanf(p, "%*[^:]:%d", &cfg.d_state);
    if ((p = strstr(json, "\"d_ff\"")))
        sscanf(p, "%*[^:]:%d", &cfg.d_ff);
    if ((p = strstr(json, "\"chunk_size\"")))
        sscanf(p, "%*[^:]:%d", &cfg.chunk_size);
    if ((p = strstr(json, "\"vocab_size\"")))
        sscanf(p, "%*[^:]:%d", &cfg.vocab_size);
    free(json);

    printf("Config: d=%d layers=%d state=%d ff=%d chunk=%d vocab=%d\n",
           cfg.d_model, cfg.n_layers, cfg.d_state, cfg.d_ff,
           cfg.chunk_size, cfg.vocab_size);

    /* Read all tensors */
    unsigned int n_tensors = read_u32(f);
    printf("Loading %u tensors...\n", n_tensors);

    NamedTensor *tensors = (NamedTensor *)malloc(n_tensors * sizeof(NamedTensor));
    for (unsigned int i = 0; i < n_tensors; i++)
    {
        unsigned int name_len = read_u32(f);
        fread(tensors[i].name, 1, name_len, f);
        tensors[i].name[name_len] = 0;

        int ndim = (int)read_u32(f);
        int shape[4] = {1, 1, 1, 1};
        int total = 1;
        for (int d = 0; d < ndim; d++)
        {
            shape[d] = (int)read_u32(f);
            total *= shape[d];
        }
        float *data = (float *)malloc(total * sizeof(float));
        fread(data, sizeof(float), total, f);
        tensors[i].t = load_tensor_data(data, ndim, shape);
    }
    fclose(f);

    /* Build model */
    Model *m = (Model *)calloc(1, sizeof(Model));
    m->cfg = cfg;
    int D = cfg.d_model, S = cfg.d_state, FF = cfg.d_ff;
    int L = cfg.n_layers;

    m->embed = get_tensor(tensors, n_tensors, "embed.weight");

    m->ssm = (SSMLayer *)calloc(L, sizeof(SSMLayer));
    m->ffn = (FFNLayer *)calloc(L, sizeof(FFNLayer));

    char buf[256];
    for (int l = 0; l < L; l++)
    {
        int li = l * 2; /* SSM at even indices */

        sprintf(buf, "layers.%d.in_proj.weight", li);
        m->ssm[l].in_w = get_tensor(tensors, n_tensors, buf);

        sprintf(buf, "layers.%d.conv.weight", li);
        m->ssm[l].conv_w = get_tensor(tensors, n_tensors, buf);
        sprintf(buf, "layers.%d.conv.bias", li);
        m->ssm[l].conv_b = get_tensor(tensors, n_tensors, buf);

        sprintf(buf, "layers.%d.dt_proj.weight", li);
        m->ssm[l].dt_w = get_tensor(tensors, n_tensors, buf);
        sprintf(buf, "layers.%d.dt_proj.bias", li);
        m->ssm[l].dt_b = get_tensor(tensors, n_tensors, buf);

        sprintf(buf, "layers.%d.B_proj.weight", li);
        m->ssm[l].B_w = get_tensor(tensors, n_tensors, buf);
        sprintf(buf, "layers.%d.C_proj.weight", li);
        m->ssm[l].C_w = get_tensor(tensors, n_tensors, buf);

        sprintf(buf, "layers.%d.log_A", li);
        m->ssm[l].log_A = get_tensor(tensors, n_tensors, buf);
        sprintf(buf, "layers.%d.D", li);
        m->ssm[l].D_param = get_tensor(tensors, n_tensors, buf);

        sprintf(buf, "layers.%d.out_proj.weight", li);
        m->ssm[l].out_w = get_tensor(tensors, n_tensors, buf);

        sprintf(buf, "layers.%d.norm.weight", li);
        m->ssm[l].norm_w = get_tensor(tensors, n_tensors, buf);
        sprintf(buf, "layers.%d.norm.bias", li);
        m->ssm[l].norm_b = get_tensor(tensors, n_tensors, buf);

        /* FFN at odd indices */
        int fi = l * 2 + 1;
        sprintf(buf, "layers.%d.norm.weight", fi);
        m->ffn[l].norm_w = get_tensor(tensors, n_tensors, buf);
        sprintf(buf, "layers.%d.norm.bias", fi);
        m->ffn[l].norm_b = get_tensor(tensors, n_tensors, buf);

        sprintf(buf, "layers.%d.net.0.weight", fi);
        m->ffn[l].up_w = get_tensor(tensors, n_tensors, buf);
        sprintf(buf, "layers.%d.net.0.bias", fi);
        m->ffn[l].up_b = get_tensor(tensors, n_tensors, buf);

        sprintf(buf, "layers.%d.net.3.weight", fi);
        m->ffn[l].down_w = get_tensor(tensors, n_tensors, buf);
        sprintf(buf, "layers.%d.net.3.bias", fi);
        m->ffn[l].down_b = get_tensor(tensors, n_tensors, buf);
    }

    m->final_norm_w = get_tensor(tensors, n_tensors, "norm.weight");
    m->final_norm_b = get_tensor(tensors, n_tensors, "norm.bias");

    /* Allocate buffers */
    m->ssm_state = (float **)malloc(L * sizeof(float *));
    for (int l = 0; l < L; l++)
    {
        m->ssm_state[l] = (float *)calloc(D * S, sizeof(float));
    }
    /* Conv buffer: per layer, d_model * kernel_size */
    m->conv_buf = (float *)calloc(L * D * 4, sizeof(float));
    m->x_buf = (float *)calloc(D * 2, sizeof(float));
    m->z_buf = (float *)calloc(D, sizeof(float));
    m->dt_buf = (float *)calloc(D, sizeof(float));
    m->B_buf = (float *)calloc(S, sizeof(float));
    m->C_buf = (float *)calloc(S, sizeof(float));
    m->h_buf = (float *)calloc(D * S, sizeof(float));
    m->logits = (float *)calloc(cfg.vocab_size, sizeof(float));

    printf("Model loaded: %d layers, %d dims\n", L, D);
    free(tensors); /* tensor data is kept, only index freed */
    return m;
}

/* ============================================================
   FORWARD PASS — one token at a time (streaming)
   ============================================================ */

static void forward_ssm_layer(Model *m, int layer_idx,
                              float *hidden, int D)
{
    SSMLayer *ssm = &m->ssm[layer_idx];
    int S = m->cfg.d_state;
    float *state = m->ssm_state[layer_idx];

    float *normed = (float *)alloca(D * sizeof(float));
    float *xz = (float *)alloca(D * 2 * sizeof(float));
    float *x_ssm = (float *)alloca(D * sizeof(float));
    float *z = (float *)alloca(D * sizeof(float));
    float *dt = (float *)alloca(D * sizeof(float));
    float *B = (float *)alloca(S * sizeof(float));
    float *C = (float *)alloca(S * sizeof(float));
    float *y = (float *)alloca(D * sizeof(float));
    float *out = (float *)alloca(D * sizeof(float));

    /* 1. LayerNorm */
    layernorm(normed, hidden, ssm->norm_w.data, ssm->norm_b.data, D);

    /* 2. in_proj: D → 2*D */
    matmul(xz, normed, ssm->in_w.data, D * 2, D);
    for (int i = 0; i < D; i++)
    {
        x_ssm[i] = xz[i];
        z[i] = xz[D + i];
    }

    /* 3. Depthwise conv1d (causal, kernel=4)
     *
     * Conv buffer layout: [layer][channel][4]
     * buf[0] = newest, buf[3] = oldest
     *
     * PyTorch conv1d weight shape: [channels, 1, kernel_size]
     * PyTorch does CROSS-CORRELATION (not convolution)
     * So weight[0] applies to oldest, weight[K-1] to newest
     */
    float *cbuf = m->conv_buf + (long)layer_idx * D * 4;

    /* Shift: move history back, insert new at position 0 */
    for (int i = 0; i < D; i++)
    {
        cbuf[i * 4 + 3] = cbuf[i * 4 + 2];
        cbuf[i * 4 + 2] = cbuf[i * 4 + 1];
        cbuf[i * 4 + 1] = cbuf[i * 4 + 0];
        cbuf[i * 4 + 0] = x_ssm[i];
    }

    /* Conv weight may have shape [D, 1, 4] — stored as D*4 floats
     * PyTorch cross-correlation:
     *   out[i] = sum_k weight[i][0][k] * input[t - (K-1) + k]
     *   weight[0] * oldest + weight[3] * newest
     */
    int conv_stride = (ssm->conv_w.ndim == 3) ? ssm->conv_w.shape[1] * ssm->conv_w.shape[2] : ssm->conv_w.shape[1];
    int K = (ssm->conv_w.ndim == 3) ? ssm->conv_w.shape[2] : ssm->conv_w.shape[1];
    if (K > 4)
        K = 4;

    for (int i = 0; i < D; i++)
    {
        float sum = ssm->conv_b.data[i];
        /* cbuf[i*4 + 0] = newest (t)
         * cbuf[i*4 + 1] = t-1
         * cbuf[i*4 + 2] = t-2
         * cbuf[i*4 + 3] = oldest (t-3)
         *
         * Cross-correlation: weight[k] * input[t - (K-1) + k]
         * weight[0] * cbuf[3], weight[1] * cbuf[2],
         * weight[2] * cbuf[1], weight[3] * cbuf[0]
         */
        for (int k = 0; k < K; k++)
        {
            sum += ssm->conv_w.data[i * conv_stride + k] *
                   cbuf[i * 4 + (K - 1 - k)];
        }
        x_ssm[i] = silu(sum);
    }

    /* 4. dt, B, C projections */
    matmul_bias(dt, x_ssm, ssm->dt_w.data, ssm->dt_b.data, D, D);
    for (int i = 0; i < D; i++)
        dt[i] = softplus(dt[i]);

    matmul(B, x_ssm, ssm->B_w.data, S, D);
    matmul(C, x_ssm, ssm->C_w.data, S, D);

    /* 5. SSM step */
    for (int i = 0; i < D; i++)
    {
        float A = -expf(ssm->log_A.data[i]);
        float decay = expf(dt[i] * A);
        float y_i = 0;
        for (int s = 0; s < S; s++)
        {
            int idx = i * S + s;
            state[idx] = state[idx] * decay + dt[i] * B[s] * x_ssm[i];
            y_i += state[idx] * C[s];
        }
        y[i] = y_i + x_ssm[i] * ssm->D_param.data[i];
    }

    /* 6. Gate and project */
    for (int i = 0; i < D; i++)
        y[i] = y[i] * silu(z[i]);

    matmul(out, y, ssm->out_w.data, D, D);

    /* 7. Residual */
    for (int i = 0; i < D; i++)
        hidden[i] += out[i];
}

static void forward_ffn_layer(Model *m, int layer_idx,
                              float *hidden, int D)
{
    FFNLayer *ffn = &m->ffn[layer_idx];
    int FF = m->cfg.d_ff;

    float *normed = (float *)alloca(D * sizeof(float));
    float *up = (float *)alloca(FF * sizeof(float));
    float *down = (float *)alloca(D * sizeof(float));

    layernorm(normed, hidden, ffn->norm_w.data, ffn->norm_b.data, D);
    matmul_bias(up, normed, ffn->up_w.data, ffn->up_b.data, FF, D);
    for (int i = 0; i < FF; i++)
        up[i] = gelu(up[i]);
    matmul_bias(down, up, ffn->down_w.data, ffn->down_b.data, D, FF);
    for (int i = 0; i < D; i++)
        hidden[i] += down[i];
}

static int forward_token(Model *m, int token)
{
    int D = m->cfg.d_model;
    int V = m->cfg.vocab_size;
    int L = m->cfg.n_layers;

    float *hidden = (float *)alloca(D * sizeof(float));

    /* Embedding */
    for (int i = 0; i < D; i++)
        hidden[i] = m->embed.data[token * D + i];

    /* Layers */
    for (int l = 0; l < L; l++)
    {
        forward_ssm_layer(m, l, hidden, D);
        forward_ffn_layer(m, l, hidden, D);
    }

    /* Final norm */
    float *normed = (float *)alloca(D * sizeof(float));
    layernorm(normed, hidden, m->final_norm_w.data,
              m->final_norm_b.data, D);

    /* Head (= embed^T) */
    matmul(m->logits, normed, m->embed.data, V, D);

    return 0;
}

/* ============================================================
   SAMPLING
   ============================================================ */

static int sample_token(float *logits, int V, float temp)
{
    if (temp < 0.01f)
    {
        /* Greedy */
        int best = 0;
        for (int i = 1; i < V; i++)
            if (logits[i] > logits[best])
                best = i;
        return best;
    }

    /* Temperature */
    float max_l = logits[0];
    for (int i = 1; i < V; i++)
        if (logits[i] > max_l)
            max_l = logits[i];

    float sum = 0;
    for (int i = 0; i < V; i++)
    {
        logits[i] = expf((logits[i] - max_l) / temp);
        sum += logits[i];
    }
    for (int i = 0; i < V; i++)
        logits[i] /= sum;

    /* Multinomial */
    float r = (float)rand() / RAND_MAX;
    float cum = 0;
    for (int i = 0; i < V; i++)
    {
        cum += logits[i];
        if (cum > r)
            return i;
    }
    return V - 1;
}

/* ============================================================
   STATE MANAGEMENT
   ============================================================ */

static void reset_state(Model *m)
{
    int L = m->cfg.n_layers;
    int D = m->cfg.d_model;
    int S = m->cfg.d_state;
    for (int l = 0; l < L; l++)
        memset(m->ssm_state[l], 0, D * S * sizeof(float));
    memset(m->conv_buf, 0, L * D * 4 * sizeof(float));
}

/* ============================================================
   UTF-8 OUTPUT
   ============================================================ */

static void print_utf8_byte(unsigned char b, unsigned char *utf8_buf,
                            int *utf8_pos, int *utf8_need)
{
    if (*utf8_need == 0)
    {
        /* Start new character */
        if (b < 0x80)
        {
            putchar(b);
            fflush(stdout);
            return;
        }
        else if ((b & 0xE0) == 0xC0)
        {
            *utf8_need = 2;
        }
        else if ((b & 0xF0) == 0xE0)
        {
            *utf8_need = 3;
        }
        else if ((b & 0xF8) == 0xF0)
        {
            *utf8_need = 4;
        }
        else
        {
            return; /* invalid */
        }
        utf8_buf[0] = b;
        *utf8_pos = 1;
    }
    else
    {
        utf8_buf[*utf8_pos] = b;
        (*utf8_pos)++;
    }

    if (*utf8_pos == *utf8_need)
    {
        fwrite(utf8_buf, 1, *utf8_need, stdout);
        fflush(stdout);
        *utf8_need = 0;
        *utf8_pos = 0;
    }
}

/* ============================================================
   GENERATE
   ============================================================ */

static void generate(Model *m, const char *prompt, int max_tokens,
                     float temp)
{
    int len = strlen(prompt);
    unsigned char utf8_buf[4];
    int utf8_pos = 0, utf8_need = 0;

    reset_state(m);

    /* Feed prompt */
    printf("\033[1;36m%s\033[0m", prompt); /* cyan prompt */
    fflush(stdout);

    int last_token = 0;
    for (int i = 0; i < len; i++)
    {
        last_token = (unsigned char)prompt[i];
        forward_token(m, last_token);
    }

    /* Generate */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < max_tokens; i++)
    {
        int tok = sample_token(m->logits, m->cfg.vocab_size, temp);
        print_utf8_byte((unsigned char)tok, utf8_buf,
                        &utf8_pos, &utf8_need);
        forward_token(m, tok);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) +
                     (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double tps = max_tokens / elapsed;

    printf("\n\033[0;33m[%d tok, %.1fs, %.0f tok/s, temp=%.2f]\033[0m\n",
           max_tokens, elapsed, tps, temp);
}

/* ============================================================
   INTERACTIVE CHAT
   ============================================================ */

static void chat_mode(Model *m)
{
    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   MAMBA-2 INTERACTIVE CHAT               ║\n");
    printf("║                                          ║\n");
    printf("║   Введите текст — модель продолжит.      ║\n");
    printf("║   /temp 0.5  — температура               ║\n");
    printf("║   /len 200   — длина генерации           ║\n");
    printf("║   /reset     — сброс состояния           ║\n");
    printf("║   /quit      — выход                     ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    float temp = 0.7f;
    int gen_len = 200;
    char input[4096];

    while (1)
    {
        printf("\033[1;32m▶ \033[0m");
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin))
            break;

        /* Remove trailing newline */
        int len = strlen(input);
        while (len > 0 && (input[len - 1] == '\n' || input[len - 1] == '\r'))
            input[--len] = 0;

        if (len == 0)
            continue;

        /* Commands */
        if (strcmp(input, "/quit") == 0 || strcmp(input, "/q") == 0)
            break;

        if (strcmp(input, "/reset") == 0)
        {
            reset_state(m);
            printf("  Состояние сброшено.\n\n");
            continue;
        }

        if (strncmp(input, "/temp", 5) == 0)
        {
            if (len > 6)
            {
                temp = atof(input + 6);
                printf("  Температура: %.2f\n\n", temp);
            }
            else
            {
                printf("  Текущая: %.2f\n\n", temp);
            }
            continue;
        }

        if (strncmp(input, "/len", 4) == 0)
        {
            if (len > 5)
            {
                gen_len = atoi(input + 5);
                printf("  Длина: %d\n\n", gen_len);
            }
            else
            {
                printf("  Текущая: %d\n\n", gen_len);
            }
            continue;
        }

        /* Generate */
        generate(m, input, gen_len, temp);
        printf("\n");
    }

    printf("Выход.\n");
}

/* ============================================================
   MAIN
   ============================================================ */

int main(int argc, char **argv)
{
    srand(time(NULL));

    if (argc < 2)
    {
        printf("Mamba-2 Inference Engine v3\n\n");
        printf("Использование:\n");
        printf("  %s model.bin \"text\" 200 0.7  — генерация\n", argv[0]);
        printf("  %s model.bin --chat            — чат\n", argv[0]);
        printf("  %s model.bin --test            — тест загрузки\n", argv[0]);
        return 1;
    }

    printf("Загрузка модели: %s\n", argv[1]);
    Model *m = load_model(argv[1]);
    if (!m)
    {
        fprintf(stderr, "Ошибка загрузки модели!\n");
        return 1;
    }

    /* Test mode — verify weights */
    if (argc >= 3 && strcmp(argv[2], "--test") == 0)
    {
        printf("\n=== ТЕСТ ЗАГРУЗКИ ===\n");
        printf("Embed[0][0..4]: ");
        for (int i = 0; i < 4; i++)
            printf("%.4f ", m->embed.data[i]);
        printf("\n");
        printf("Conv[0] shape: ndim=%d", m->ssm[0].conv_w.ndim);
        for (int i = 0; i < m->ssm[0].conv_w.ndim; i++)
            printf(" [%d]", m->ssm[0].conv_w.shape[i]);
        printf(" total=%d\n", m->ssm[0].conv_w.size);
        printf("log_A[0][0..4]: ");
        for (int i = 0; i < 4; i++)
            printf("%.4f ", m->ssm[0].log_A.data[i]);
        printf("\n");

        /* Quick generation test */
        printf("\nТест генерации (5 токенов, temp=0):\n");
        reset_state(m);
        int tok = 'A';
        for (int i = 0; i < 5; i++)
        {
            forward_token(m, tok);
            tok = sample_token(m->logits, m->cfg.vocab_size, 0.0f);
            printf("  %d -> %d ('%c')\n", i, tok,
                   (tok >= 32 && tok < 127) ? tok : '?');
        }
        printf("\n");

        /* Russian test */
        printf("Тест русского (temp=0.3):\n  ");
        generate(m, "Россия ", 50, 0.3f);
        printf("\nТест английского (temp=0.3):\n  ");
        generate(m, "The ", 50, 0.3f);
        printf("\n");
        return 0;
    }

    /* Chat mode */
    if (argc >= 3 && strcmp(argv[2], "--chat") == 0)
    {
        chat_mode(m);
        return 0;
    }

    /* Generation mode */
    const char *prompt = (argc >= 3) ? argv[2] : "Привет";
    int max_tokens = (argc >= 4) ? atoi(argv[3]) : 200;
    float temp = (argc >= 5) ? atof(argv[4]) : 0.7f;

    printf("Prompt: \"%s\" | Tokens: %d | Temp: %.2f\n\n",
           prompt, max_tokens, temp);

    generate(m, prompt, max_tokens, temp);
    printf("\n");

    return 0;
}