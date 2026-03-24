#!/usr/bin/env python3
"""Simple interactive chat with any configured model via LiteLLM.

Usage:
  python3 chat.py                  # freeform chat
  python3 chat.py --conversion     # use the conversion task system prompt
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "1_run_inference"))

from config import MODEL_CONFIGS, setup_api_keys, get_system_prompt
import litellm

litellm.suppress_debug_info = True

def main():
    parser = argparse.ArgumentParser(description="Chat with any configured model")
    parser.add_argument("--conversion", action="store_true",
                        help="Use the conversion task system prompt")
    args = parser.parse_args()

    setup_api_keys()

    names = sorted(MODEL_CONFIGS.keys())
    print("Available models:")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")

    choice = input("\nPick a model (name or number): ").strip()
    if choice.isdigit():
        model_name = names[int(choice) - 1]
    elif choice in MODEL_CONFIGS:
        model_name = choice
    else:
        print(f"Unknown model: {choice}")
        return

    config = MODEL_CONFIGS[model_name]
    is_reasoning = config.get("reasoning", False)

    history = []
    if args.conversion:
        sys_prompt = get_system_prompt(is_timezone=False, is_reasoning=is_reasoning)
        history.append({"role": "system", "content": sys_prompt})
        print(f"\n[System prompt loaded: conversion mode, reasoning={is_reasoning}]")

    print(f"Chatting with {model_name} ({config['litellm_model']})")
    print("Type your message. 'q' to quit, 'clear' to reset history.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() == "q":
            break
        if user_input.lower() == "clear":
            history.clear()
            print("-- history cleared --\n")
            continue

        history.append({"role": "user", "content": user_input})

        params = {
            "model": config["litellm_model"],
            "messages": history,
            "stream": False,
        }
        if not config.get("reasoning", False):
            params["temperature"] = 0.7
            params["max_tokens"] = 2048
        else:
            params["max_tokens"] = 16000
        if "extra_body" in config:
            params["extra_body"] = config["extra_body"]
        if "extra_params" in config:
            params.update(config["extra_params"])
        if "timeout" in config:
            params["timeout"] = config["timeout"]

        try:
            resp = litellm.completion(**params)
            content = resp.choices[0].message.content or ""
            print(f"\n{model_name}: {content}\n")
            history.append({"role": "assistant", "content": content})
        except Exception as e:
            print(f"\nERROR: {e}\n")
            history.pop()

if __name__ == "__main__":
    main()
