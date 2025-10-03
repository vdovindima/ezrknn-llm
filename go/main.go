package main

/*
#cgo CFLAGS: -I. -std=c99
#cgo LDFLAGS: -L. -lwrapper -lrkllm
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "rkllm.h"

// Forward declaration of Go callback so C can call it.
extern int goCallback(RKLLMResult* result, void* userdata, LLMCallState state);

// Declarations for functions implemented in wrapper.c
int c_callback(RKLLMResult* result, void* userdata, LLMCallState state);
int set_rkllm_input_prompt(RKLLMInput* in, const char* role, const char* prompt);
*/
import "C"

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"time"
	"unsafe"
)

var llmHandle C.LLMHandle
var lastPerfStat C.RKLLMPerfStat

//export goCallback
func goCallback(result *C.RKLLMResult, userdata unsafe.Pointer, state C.LLMCallState) C.int {
	if state == C.RKLLM_RUN_FINISH {
		lastPerfStat = result.perf
		fmt.Println()
	} else if state == C.RKLLM_RUN_ERROR {
		fmt.Println("\\run error")
	} else if state == C.RKLLM_RUN_NORMAL {
		h := result.last_hidden_layer
		if h.embd_size != 0 && h.num_tokens != 0 && h.hidden_states != nil {
			dataSize := int(h.embd_size) * int(h.num_tokens) * int(unsafe.Sizeof(float32(0)))
			fmt.Printf("\ndata_size:%d", dataSize)
			data := C.GoBytes(unsafe.Pointer(h.hidden_states), C.int(dataSize))
			if err := ioutil.WriteFile("last_hidden_layer.bin", data, 0644); err == nil {
				fmt.Println("\nData saved to last_hidden_layer.bin successfully!")
			} else {
				fmt.Fprintln(os.Stderr, "Failed to write last_hidden_layer.bin:", err)
			}
		}
		if result.text != nil {
			fmt.Print(C.GoString(result.text))
		}
	}
	return 0
}

func printUsage(prog string) {
	fmt.Printf("Usage: %s model_path max_new_tokens max_context_len [top_k] [top_p] [temperature] [repeat_penalty] [frequency_penalty] [presence_penalty] [skip_special_token(0/1)] [base_domain_id] [embed_flash(0/1)]\n\n", prog)
	fmt.Println("Defaults:")
	fmt.Println("  top_k = 1")
	fmt.Println("  top_p = 0.95")
	fmt.Println("  temperature = 0.8")
	fmt.Println("  repeat_penalty = 1.1")
	fmt.Println("  frequency_penalty = 0.0")
	fmt.Println("  presence_penalty = 0.0")
	fmt.Println("  skip_special_token = 1")
	fmt.Println("  base_domain_id = 0")
	fmt.Println("  embed_flash = 1")
}

func exitHandler() {
	if llmHandle != nil {
		fmt.Println("Program is about to exit...")
		tmp := llmHandle
		llmHandle = nil
		C.rkllm_destroy(tmp)
	}
}

func main() {
	if len(os.Args) < 4 {
		printUsage(os.Args[0])
		os.Exit(1)
	}

	// handle SIGINT
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	go func() {
		<-sigCh
		exitHandler()
		os.Exit(0)
	}()

	fmt.Println("RKLLM starting, please wait...")

	// Required args
	modelPath := os.Args[1]
	maxNewTokens, _ := strconv.Atoi(os.Args[2])
	maxContextLen, _ := strconv.Atoi(os.Args[3])

	// Optional args with defaults
	topK := 1
	topP := 0.95
	temperature := 0.8
	repeatPenalty := 1.1
	frequencyPenalty := 0.0
	presencePenalty := 0.0
	skipSpecialToken := 1
	baseDomainId := 0
	embedFlash := 1

	if len(os.Args) >= 5 {
		if v, err := strconv.Atoi(os.Args[4]); err == nil {
			topK = v
		}
	}
	if len(os.Args) >= 6 {
		if v, err := strconv.ParseFloat(os.Args[5], 64); err == nil {
			topP = v
		}
	}
	if len(os.Args) >= 7 {
		if v, err := strconv.ParseFloat(os.Args[6], 64); err == nil {
			temperature = v
		}
	}
	if len(os.Args) >= 8 {
		if v, err := strconv.ParseFloat(os.Args[7], 64); err == nil {
			repeatPenalty = v
		}
	}
	if len(os.Args) >= 9 {
		if v, err := strconv.ParseFloat(os.Args[8], 64); err == nil {
			frequencyPenalty = v
		}
	}
	if len(os.Args) >= 10 {
		if v, err := strconv.ParseFloat(os.Args[9], 64); err == nil {
			presencePenalty = v
		}
	}
	if len(os.Args) >= 11 {
		if v, err := strconv.Atoi(os.Args[10]); err == nil {
			skipSpecialToken = v
		}
	}
	if len(os.Args) >= 12 {
		if v, err := strconv.Atoi(os.Args[11]); err == nil {
			baseDomainId = v
		}
	}
	if len(os.Args) >= 13 {
		if v, err := strconv.Atoi(os.Args[12]); err == nil {
			embedFlash = v
		}
	}

	// Basic validation
	if maxNewTokens < 0 {
		maxNewTokens = 128
	}
	if maxContextLen <= 0 {
		maxContextLen = 2048
	}
	if topK < 1 {
		topK = 1
	}
	if topP < 0.0 || topP > 1.0 {
		topP = 0.95
	}
	if temperature <= 0.0 {
		temperature = 0.8
	}
	if repeatPenalty <= 0.0 {
		repeatPenalty = 1.0
	}
	if skipSpecialToken != 0 && skipSpecialToken != 1 {
		skipSpecialToken = 1
	}
	if embedFlash != 0 && embedFlash != 1 {
		embedFlash = 1
	}

	// Set parameters and initialize
	param := C.rkllm_createDefaultParam()
	cModelPath := C.CString(modelPath)
	param.model_path = cModelPath
	// ensure we free model path after init
	defer C.free(unsafe.Pointer(cModelPath))

	// sampling params (C floats)
	param.top_k = C.int(topK)
	param.top_p = C.float(topP)
	param.temperature = C.float(temperature)
	param.repeat_penalty = C.float(repeatPenalty)
	param.frequency_penalty = C.float(frequencyPenalty)
	param.presence_penalty = C.float(presencePenalty)

	param.max_new_tokens = C.int(maxNewTokens)
	param.max_context_len = C.int(maxContextLen)
	param.skip_special_token = C.bool(skipSpecialToken != 0)
	// extend param assignments
	param.extend_param.base_domain_id = C.int32_t(baseDomainId)
	param.extend_param.embed_flash = C.int8_t(embedFlash)

	// Initialize with C callback wrapper (cast to expected callback type)
	ret := C.rkllm_init(&llmHandle, &param, (C.LLMResultCallback)(C.c_callback))
	if ret == 0 {
		fmt.Println("rkllm init success")
	} else {
		fmt.Println("rkllm init failed")
		exitHandler()
		os.Exit(1)
	}

	preInput := []string{
		"Welcome to ezrkllm! This is an adaptation of Rockchip's rknn-llm repo (see github.com/airockchip/rknn-llm) for running LLMs on its SoCs' NPUs.\n",
		"To exit the model, enter either exit or quit\n",
		"More information here: https://github.com/Pelochus/ezrknpu\n",
		"Detailed information for devs here: https://github.com/Pelochus/ezrknn-llm\n",
	}

	fmt.Println("\n*************************** Pelochus' ezrkllm runtime *************************\n")
	for i, v := range preInput {
		fmt.Printf("[%d] %s", i, v)
	}
	fmt.Println("\n*************************************************************************\n")

	var rkllmInput C.RKLLMInput
	var rkllmInferParams C.RKLLMInferParam
	rkllmInferParams.mode = C.RKLLM_INFER_GENERATE
	rkllmInferParams.keep_history = 0

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\nYou: ")
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		inputStr := strings.TrimRight(line, "\r\n")

		if inputStr == "exit" || inputStr == "quit" {
			fmt.Println("Quitting rkllm...")
			break
		}
		if inputStr == "clear" {
			cr := C.rkllm_clear_kv_cache(llmHandle, 1, nil, nil)
			if cr != 0 {
				fmt.Println("clear kv cache failed!")
			}
			continue
		}
		for i := range preInput {
			if inputStr == strconv.Itoa(i) {
				inputStr = preInput[i]
				fmt.Println(inputStr)
			}
		}

		// prepare C strings for role and prompt
		cRole := C.CString("User")
		cPrompt := C.CString(inputStr)

		// set input via helper to avoid union access issues
		C.set_rkllm_input_prompt(&rkllmInput, cRole, cPrompt)

		fmt.Print("LLM: ")
		start := time.Now()
		C.rkllm_run(llmHandle, &rkllmInput, &rkllmInferParams, nil)
		elapsed := time.Since(start)

		// free prompt/role C strings
		C.free(unsafe.Pointer(cRole))
		C.free(unsafe.Pointer(cPrompt))

		tokenCount := uint32(lastPerfStat.prefill_tokens) + uint32(lastPerfStat.generate_tokens)
		if tokenCount > 0 && elapsed.Seconds() > 0.0 {
			tokensPerSec := float64(tokenCount) / elapsed.Seconds()
			fmt.Printf("\n[Token/s]: %.2f\n[Tokens]: %d\n[Seconds]: %.3f\n", tokensPerSec, tokenCount, elapsed.Seconds())
		}
	}

	C.rkllm_destroy(llmHandle)
}
