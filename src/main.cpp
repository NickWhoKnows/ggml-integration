#include "gguf.h"


int main()
{

    {
        char *pth = (char *)"../models/Llama-3.2-1B-Instruct.gguf";
        GGUF gguf(pth);
        gguf.printHeader();
        gguf.printMetadata();
        gguf.printTensors();
        gguf.printTensorData("blk.0.attn_norm.weight");
        gguf.printTensorData("token_embd.weight", 16);
    }

    return 0;
}