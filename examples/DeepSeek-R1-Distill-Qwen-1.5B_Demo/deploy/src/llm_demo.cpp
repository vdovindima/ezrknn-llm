// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified by Pelochus, adapted to accept parameters via argv only.

#include <string.h>
#include <unistd.h>
#include <string>
#include "rkllm.h"
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>

using namespace std;

LLMHandle llmHandle = nullptr;

void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        cout << "Program is about to exit..." << endl;
        LLMHandle _tmp = llmHandle;
        llmHandle = nullptr;
        rkllm_destroy(_tmp);
    }

    exit(signal);
}

// Get perf stats (token/s)
RKLLMPerfStat last_perf_stat = {0};
int callback(RKLLMResult *result, void *userdata, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH) {
        last_perf_stat = result->perf; // Store perf stats
        printf("\n");
    } else if (state == RKLLM_RUN_ERROR) {
        printf("\\run error\n");
    } else if (state == RKLLM_RUN_NORMAL) {
        /*
        ======================================================================================================================
        When using the GET_LAST_HIDDEN_LAYER function, the callback interface will return a memory pointer: last_hidden_layer,
        the number of tokens: num_tokens, and the hidden layer size: embd_size.
        These three parameters can be used to access the data in last_hidden_layer.

        Note: You must retrieve the data within the current callback; if not retrieved in time, the pointer will be released
        during the next callback.
        ======================================================================================================================
        */

        if (result->last_hidden_layer.embd_size != 0 && result->last_hidden_layer.num_tokens != 0) {
            int data_size = result->last_hidden_layer.embd_size * result->last_hidden_layer.num_tokens * sizeof(float);
            printf("\ndata_size:%d",data_size);
            ofstream outFile("last_hidden_layer.bin", ios::binary);

            if (outFile.is_open()) {
                outFile.write(reinterpret_cast<const char*>(result->last_hidden_layer.hidden_states), data_size);
                outFile.close();
                cout << "Data saved to last_hidden_layer.bin successfully!" << endl;
            } else {
                cerr << "Failed to open the file for writing!" << endl;
            }
        }

        printf("%s", result->text);
    }
    return 0;
}

static void print_usage(const char *prog)
{
    cout << "Usage: " << prog << " model_path max_new_tokens max_context_len"
         << " [top_k] [top_p] [temperature] [repeat_penalty] [frequency_penalty] [presence_penalty]"
         << " [skip_special_token(0/1)] [base_domain_id] [embed_flash(0/1)]\n\n";

    cout << "Defaults:\n"
         << "  top_k = 1\n"
         << "  top_p = 0.95\n"
         << "  temperature = 0.8\n"
         << "  repeat_penalty = 1.1\n"
         << "  frequency_penalty = 0.0\n"
         << "  presence_penalty = 0.0\n"
         << "  skip_special_token = 1\n"
         << "  base_domain_id = 0\n"
         << "  embed_flash = 1\n\n";
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    signal(SIGINT, exit_handler);
    printf("RKLLM starting, please wait...\n");

    // Required args
    string model_path = argv[1];
    int max_new_tokens = atoi(argv[2]);
    int max_context_len = atoi(argv[3]);

    // Optional args with defaults
    int top_k = 1;
    double top_p = 0.95;
    double temperature = 0.8;
    double repeat_penalty = 1.1;
    double frequency_penalty = 0.0;
    double presence_penalty = 0.0;
    int skip_special_token = 1;
    int base_domain_id = 0;
    int embed_flash = 1;

    if (argc >= 5) top_k = atoi(argv[4]);
    if (argc >= 6) top_p = atof(argv[5]);
    if (argc >= 7) temperature = atof(argv[6]);
    if (argc >= 8) repeat_penalty = atof(argv[7]);
    if (argc >= 9) frequency_penalty = atof(argv[8]);
    if (argc >= 10) presence_penalty = atof(argv[9]);
    if (argc >= 11) skip_special_token = atoi(argv[10]);
    if (argc >= 12) base_domain_id = atoi(argv[11]);
    if (argc >= 13) embed_flash = atoi(argv[12]);

    // Validate simple ranges (best-effort)
    if (max_new_tokens < 0) max_new_tokens = 128;
    if (max_context_len <= 0) max_context_len = 2048;
    if (top_k < 1) top_k = 1;
    if (top_p < 0.0 || top_p > 1.0) top_p = 0.95;
    if (temperature <= 0.0) temperature = 0.8;
    if (repeat_penalty <= 0.0) repeat_penalty = 1.0;
    if (skip_special_token != 0 && skip_special_token != 1) skip_special_token = 1;
    if (embed_flash != 0 && embed_flash != 1) embed_flash = 1;

    // Set parameters and initialize
    RKLLMParam param = rkllm_createDefaultParam();
    // Ensure the std::string outlives usage (it does in this scope)
    param.model_path = model_path.c_str();

    // Set sampling parameters from argv
    param.top_k = top_k;
    param.top_p = top_p;
    param.temperature = temperature;
    param.repeat_penalty = repeat_penalty;
    param.frequency_penalty = frequency_penalty;
    param.presence_penalty = presence_penalty;

    param.max_new_tokens = max_new_tokens;
    param.max_context_len = max_context_len;
    param.skip_special_token = skip_special_token ? 1 : 0;
    param.extend_param.base_domain_id = base_domain_id;
    param.extend_param.embed_flash = embed_flash;

    int ret = rkllm_init(&llmHandle, &param, callback);

    if (ret == 0){
        printf("rkllm init success\n");
    } else {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }

    vector<string> pre_input;
    pre_input.push_back("Welcome to ezrkllm! This is an adaptation of Rockchip's rknn-llm repo (see github.com/airockchip/rknn-llm) for running LLMs on its SoCs' NPUs.\n");
    pre_input.push_back("To exit the model, enter either exit or quit\n");
    pre_input.push_back("More information here: https://github.com/Pelochus/ezrknpu\n");
    pre_input.push_back("Detailed information for devs here: https://github.com/Pelochus/ezrknn-llm\n");

    cout << "\n*************************** Pelochus' ezrkllm runtime *************************\n" << endl;

    for (int i = 0; i < (int) pre_input.size(); i++)
    {
        cout << "[" << i << "] " << pre_input[i];
    }

    cout << "\n*************************************************************************\n" << endl;

    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    // Initialize the infer parameter structure
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));  // Initialize all fields to 0

    // (LoRA and prompt cache code remains commented as in original; user can enable if needed)

    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    // By default, the chat operates in single-turn mode (no context retention)
    // 0 means no history is retained, each query is independent
    rkllm_infer_params.keep_history = 0;

    while (true)
    {
        string input_str;
        printf("\n");
        printf("You: ");
        if (!getline(cin, input_str)) break; // handle EOF

        if (input_str == "exit" || input_str == "quit")
        {
            cout << "Quitting rkllm..." << endl;
            break;
        }

        if (input_str == "clear")
        {
            ret = rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
            if (ret != 0)
            {
                printf("clear kv cache failed!\n");
            }

            continue;
        }

        for (int i = 0; i < (int)pre_input.size(); i++)
        {
            if (input_str == to_string(i))
            {
                input_str = pre_input[i];
                cout << input_str << endl;
            }
        }

        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.role = "User";
        rkllm_input.prompt_input = (char*) input_str.c_str();
        printf("LLM: ");

        const auto start_time = chrono::steady_clock::now();
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, nullptr);
        const auto end_time = chrono::steady_clock::now();

        const chrono::duration<double> elapsed = end_time - start_time;

        const uint32_t token_count = last_perf_stat.prefill_tokens + last_perf_stat.generate_tokens;
        if (token_count > 0 && elapsed.count() > 0.0) {
            const double tokens_per_sec = token_count / elapsed.count();
            cout << "\n[Token/s]: " << fixed << setprecision(2) << tokens_per_sec
                 << "\n[Tokens]: "  << token_count
                 << "\n[Seconds]: " << elapsed.count() << endl;
        }
    }

    rkllm_destroy(llmHandle);

    return 0;
}
