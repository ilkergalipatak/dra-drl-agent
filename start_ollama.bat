@echo off
echo Starting Ollama Server with Optimization Config...
set OLLAMA_HOST=0.0.0.0
set OLLAMA_NUM_PARALLEL=4
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_NUM_CTX=262144
ollama serve
