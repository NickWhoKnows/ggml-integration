#include <ggml.h>
#include <ggml-cpu.h>
#include <iostream>


int main()
{

    const int r = 2, h = 2;
    const float m1[r * h] = {
        1, 2,
        3, 4};
    const float m2[r * h] = {
        5, 6,
        7, 8};

    size_t ctx_size = 0;

    ctx_size += r * h * ggml_type_size(GGML_TYPE_F32);
    ctx_size += r * h * ggml_type_size(GGML_TYPE_F32);
    ctx_size += r * h * ggml_type_size(GGML_TYPE_F32);
    ctx_size += 3 * ggml_tensor_overhead(); // metadata for 3 tensors
    ctx_size += ggml_graph_overhead();      // compute graph
    ctx_size += 1024;                       // some overhead (exact calculation omitted for simplicity)

    std::cout << "Estimated context size: " << ctx_size << " bytes" << std::endl;

    struct ggml_init_params params = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };

    struct ggml_context *ctx = ggml_init(params);
    struct ggml_tensor *tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, h);
    struct ggml_tensor *tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, h);

    memcpy(tensor_a->data, m1, ggml_nbytes(tensor_a));
    memcpy(tensor_b->data, m2, ggml_nbytes(tensor_b));

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    struct ggml_tensor *result = ggml_mul_mat(ctx, tensor_a, tensor_b);
    ggml_build_forward_expand(gf, result);
    
    int n_threads = 1; // Optional: number of threads to perform some operations with multi-threading
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    
    float *result_data = (float *)result->data;
    printf("mul mat (%d x %d) (transposed result):\n[", (int)result->ne[0], (int)result->ne[1]);

    for (int j = 0; j < result->ne[1] /* rows */; j++){
        if (j > 0){
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++){
            printf(" %.2f", result_data[j * result->ne[0] + i]);
        }
    }

    printf(" ]\n");
    ggml_free(ctx);
    return 0;
}